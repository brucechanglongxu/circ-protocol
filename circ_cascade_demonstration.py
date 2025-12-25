"""
CIRC Cascade Failure Demonstration: Error Propagation and Containment

This simulation demonstrates how errors propagate through multi-agent systems
and how CIRC infrastructure enables early detection and containment.

Key insight: In L1 (isolated agents), errors are invisible to other agents
and cascade unchecked until they manifest as patient harm. Under L3/L4,
the coordination infrastructure provides checkpoints that catch errors
before they propagate.

Clinical scenario modeled:
1. Initial error: Lab processing agent misreads a critical value
2. Downstream agents act on incorrect information
3. Cascade: Wrong treatment scheduled → wrong medications ordered → 
   conflicting care plans → patient harm

This connects to Asimov's Laws framing:
- L0: "Do no harm" - but how do you prevent harm from cascading errors?
- CIRC provides the infrastructure for error detection and containment

From paper Section 3.2:
"Reversibility constraints ensure any automated action can be reversed 
within defined windows if errors are detected downstream."
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Tuple
from enum import Enum
from collections import defaultdict
import json

random.seed(42)
np.random.seed(42)

# =============================================================================
# Data Structures
# =============================================================================

class ErrorType(Enum):
    LAB_MISREAD = "lab_misread"
    WRONG_PATIENT = "wrong_patient"
    MISSED_CONTRAINDICATION = "missed_contraindication"
    SCHEDULING_CONFLICT = "scheduling_conflict"
    DOSAGE_ERROR = "dosage_error"

class AgentRole(Enum):
    LAB_PROCESSING = "lab_processing"
    ONCOLOGY = "oncology"
    DIABETES = "diabetes"
    PHARMACY = "pharmacy"
    SCHEDULING = "scheduling"
    NURSING = "nursing"
    RADIOLOGY = "radiology"

class ActionType(Enum):
    READ_LAB = "read_lab"
    SCHEDULE_TREATMENT = "schedule_treatment"
    ORDER_MEDICATION = "order_medication"
    APPROVE_PLAN = "approve_plan"
    EXECUTE_TREATMENT = "execute_treatment"
    UPDATE_RECORD = "update_record"

@dataclass
class ClinicalAction:
    """An action taken by an agent"""
    action_id: str
    agent_id: str
    agent_role: AgentRole
    action_type: ActionType
    patient_id: str
    timestamp: int
    input_data: dict
    output_data: dict
    based_on: List[str] = field(default_factory=list)  # Action IDs this depends on
    is_erroneous: bool = False
    error_type: Optional[ErrorType] = None
    detected: bool = False
    detection_timestamp: Optional[int] = None

@dataclass
class Patient:
    id: str
    true_platelet_count: int
    true_glucose: int
    has_diabetes: bool
    chemo_eligible: bool  # Ground truth
    
@dataclass
class CascadeEvent:
    """Tracks error propagation"""
    origin_action_id: str
    affected_action_ids: List[str]
    depth: int  # How many hops from origin
    patients_affected: Set[str]
    detected: bool
    detection_depth: Optional[int]
    harm_occurred: bool

# =============================================================================
# Agent Definitions
# =============================================================================

class BaseAgent:
    def __init__(self, agent_id: str, role: AgentRole, error_rate: float = 0.05):
        self.id = agent_id
        self.role = role
        self.error_rate = error_rate
        self.actions: List[ClinicalAction] = []
        
    def maybe_introduce_error(self) -> Tuple[bool, Optional[ErrorType]]:
        """Agents occasionally make errors"""
        if random.random() < self.error_rate:
            error_types = list(ErrorType)
            return True, random.choice(error_types)
        return False, None


class LabProcessingAgent(BaseAgent):
    """Processes lab results - first in the chain"""
    
    def __init__(self, agent_id: str, error_rate: float = 0.05):
        super().__init__(agent_id, AgentRole.LAB_PROCESSING, error_rate)
    
    def process_labs(self, patient: Patient, timestamp: int) -> ClinicalAction:
        """Process labs - may introduce errors"""
        is_error, error_type = self.maybe_introduce_error()
        
        if is_error and error_type == ErrorType.LAB_MISREAD:
            # Critical error: Misread platelet count
            # Report normal when actually low, or vice versa
            if patient.true_platelet_count < 100:
                reported_platelets = 150  # Dangerous: Report normal when low
            else:
                reported_platelets = patient.true_platelet_count
        else:
            reported_platelets = patient.true_platelet_count
            is_error = False
            error_type = None
        
        action = ClinicalAction(
            action_id=f"LAB-{patient.id}-{timestamp}",
            agent_id=self.id,
            agent_role=self.role,
            action_type=ActionType.READ_LAB,
            patient_id=patient.id,
            timestamp=timestamp,
            input_data={"true_platelets": patient.true_platelet_count},
            output_data={"reported_platelets": reported_platelets},
            is_erroneous=is_error,
            error_type=error_type
        )
        self.actions.append(action)
        return action


class OncologyAgent(BaseAgent):
    """Makes treatment decisions based on labs"""
    
    def __init__(self, agent_id: str, error_rate: float = 0.03):
        super().__init__(agent_id, AgentRole.ONCOLOGY, error_rate)
    
    def evaluate_chemo_eligibility(self, patient: Patient, lab_action: ClinicalAction, 
                                    timestamp: int) -> ClinicalAction:
        """Decide if patient can receive chemo based on reported labs"""
        reported_platelets = lab_action.output_data["reported_platelets"]
        
        # Standard protocol: Need platelets > 100 for chemo
        chemo_approved = reported_platelets >= 100
        
        is_error, error_type = self.maybe_introduce_error()
        if is_error and error_type == ErrorType.MISSED_CONTRAINDICATION:
            # Approve chemo despite contraindication
            chemo_approved = True
            
        # Check if this decision is actually erroneous
        # (approved chemo when patient truly ineligible)
        decision_erroneous = (chemo_approved and not patient.chemo_eligible)
        
        # Inherit error from upstream if we're acting on bad data
        inherited_error = lab_action.is_erroneous
        
        action = ClinicalAction(
            action_id=f"ONCO-{patient.id}-{timestamp}",
            agent_id=self.id,
            agent_role=self.role,
            action_type=ActionType.APPROVE_PLAN,
            patient_id=patient.id,
            timestamp=timestamp,
            input_data={"reported_platelets": reported_platelets},
            output_data={"chemo_approved": chemo_approved},
            based_on=[lab_action.action_id],
            is_erroneous=decision_erroneous or inherited_error,
            error_type=ErrorType.LAB_MISREAD if inherited_error else (error_type if is_error else None)
        )
        self.actions.append(action)
        return action


class PharmacyAgent(BaseAgent):
    """Orders medications based on treatment plan"""
    
    def __init__(self, agent_id: str, error_rate: float = 0.02):
        super().__init__(agent_id, AgentRole.PHARMACY, error_rate)
    
    def process_order(self, patient: Patient, onco_action: ClinicalAction,
                      timestamp: int) -> ClinicalAction:
        """Process medication order"""
        chemo_approved = onco_action.output_data["chemo_approved"]
        
        if chemo_approved:
            order = "TCHP_REGIMEN"
        else:
            order = "HOLD"
        
        is_error, error_type = self.maybe_introduce_error()
        if is_error and error_type == ErrorType.DOSAGE_ERROR:
            # Wrong dosage calculated
            order = "TCHP_REGIMEN_WRONG_DOSE"
        
        inherited_error = onco_action.is_erroneous
        
        action = ClinicalAction(
            action_id=f"PHARM-{patient.id}-{timestamp}",
            agent_id=self.id,
            agent_role=self.role,
            action_type=ActionType.ORDER_MEDICATION,
            patient_id=patient.id,
            timestamp=timestamp,
            input_data={"chemo_approved": chemo_approved},
            output_data={"order": order},
            based_on=[onco_action.action_id],
            is_erroneous=inherited_error or is_error,
            error_type=onco_action.error_type if inherited_error else (error_type if is_error else None)
        )
        self.actions.append(action)
        return action


class SchedulingAgent(BaseAgent):
    """Schedules treatments"""
    
    def __init__(self, agent_id: str, error_rate: float = 0.02):
        super().__init__(agent_id, AgentRole.SCHEDULING, error_rate)
    
    def schedule_treatment(self, patient: Patient, pharm_action: ClinicalAction,
                           timestamp: int) -> ClinicalAction:
        """Schedule the treatment"""
        order = pharm_action.output_data["order"]
        
        if order != "HOLD":
            scheduled = True
            slot = f"INFUSION-{timestamp + 1}"
        else:
            scheduled = False
            slot = None
        
        inherited_error = pharm_action.is_erroneous
        
        action = ClinicalAction(
            action_id=f"SCHED-{patient.id}-{timestamp}",
            agent_id=self.id,
            agent_role=self.role,
            action_type=ActionType.SCHEDULE_TREATMENT,
            patient_id=patient.id,
            timestamp=timestamp,
            input_data={"order": order},
            output_data={"scheduled": scheduled, "slot": slot},
            based_on=[pharm_action.action_id],
            is_erroneous=inherited_error,
            error_type=pharm_action.error_type if inherited_error else None
        )
        self.actions.append(action)
        return action


class NursingAgent(BaseAgent):
    """Executes treatments - final step where harm can occur"""
    
    def __init__(self, agent_id: str, error_rate: float = 0.01):
        super().__init__(agent_id, AgentRole.NURSING, error_rate)
    
    def execute_treatment(self, patient: Patient, sched_action: ClinicalAction,
                          timestamp: int) -> ClinicalAction:
        """Execute the treatment - this is where harm manifests"""
        scheduled = sched_action.output_data["scheduled"]
        
        if scheduled:
            executed = True
            # If error propagated here AND patient truly ineligible = HARM
            harm_occurred = sched_action.is_erroneous and not patient.chemo_eligible
        else:
            executed = False
            harm_occurred = False
        
        action = ClinicalAction(
            action_id=f"NURSE-{patient.id}-{timestamp}",
            agent_id=self.id,
            agent_role=self.role,
            action_type=ActionType.EXECUTE_TREATMENT,
            patient_id=patient.id,
            timestamp=timestamp,
            input_data={"scheduled": scheduled},
            output_data={"executed": executed, "harm_occurred": harm_occurred},
            based_on=[sched_action.action_id],
            is_erroneous=sched_action.is_erroneous,
            error_type=sched_action.error_type if sched_action.is_erroneous else None
        )
        self.actions.append(action)
        return action


# =============================================================================
# Coordination Environments
# =============================================================================

class L1Environment:
    """
    L1: No coordination infrastructure.
    Errors propagate unchecked until they manifest as harm.
    No mechanism for detection or containment.
    """
    
    def __init__(self):
        self.actions: List[ClinicalAction] = []
        self.cascades: List[CascadeEvent] = []
        
    def record_action(self, action: ClinicalAction):
        """Simply record - no checking"""
        self.actions.append(action)
    
    def check_action(self, action: ClinicalAction) -> bool:
        """L1: No checking, always passes"""
        return True
    
    def analyze_cascades(self) -> List[CascadeEvent]:
        """Post-hoc analysis of what happened"""
        # Find error origins
        error_origins = [a for a in self.actions if a.is_erroneous and not a.based_on]
        
        cascades = []
        for origin in error_origins:
            affected = self._trace_downstream(origin.action_id)
            patients = set(a.patient_id for a in affected)
            harm = any(a.output_data.get("harm_occurred", False) for a in affected)
            
            cascades.append(CascadeEvent(
                origin_action_id=origin.action_id,
                affected_action_ids=[a.action_id for a in affected],
                depth=len(affected),
                patients_affected=patients,
                detected=False,  # L1 never detects
                detection_depth=None,
                harm_occurred=harm
            ))
        
        self.cascades = cascades
        return cascades
    
    def _trace_downstream(self, action_id: str) -> List[ClinicalAction]:
        """Trace all actions affected by this one"""
        affected = []
        for action in self.actions:
            if action_id in action.based_on:
                affected.append(action)
                affected.extend(self._trace_downstream(action.action_id))
        return affected


class L3Environment:
    """
    L3: Universal coordination with consistency checking.
    
    Key capabilities:
    1. Cross-reference checks between agents
    2. Anomaly detection on action streams
    3. Reversibility windows for error correction
    
    From paper: "Boundary enforcement through whitelisted action types 
    with reversibility constraints"
    """
    
    def __init__(self, detection_probability: float = 0.7):
        self.actions: List[ClinicalAction] = []
        self.cascades: List[CascadeEvent] = []
        self.detection_probability = detection_probability
        
        # L3 coordination state
        self.action_registry: Dict[str, ClinicalAction] = {}
        self.patient_action_chains: Dict[str, List[str]] = defaultdict(list)
        self.detected_errors: List[str] = []
        self.blocked_actions: List[str] = []
        
    def record_action(self, action: ClinicalAction):
        """Record with tracking"""
        self.actions.append(action)
        self.action_registry[action.action_id] = action
        self.patient_action_chains[action.patient_id].append(action.action_id)
    
    def check_action(self, action: ClinicalAction, patient: Patient) -> Tuple[bool, Optional[str]]:
        """
        L3: Check action against coordination infrastructure.
        
        Checks performed:
        1. Consistency with upstream actions
        2. Cross-reference with original data
        3. Anomaly detection
        """
        # Check 1: Verify upstream actions exist and are valid
        for upstream_id in action.based_on:
            if upstream_id in self.detected_errors:
                # Upstream error already detected - block this action
                self.blocked_actions.append(action.action_id)
                return False, f"Blocked: upstream error detected in {upstream_id}"
        
        # Check 2: Cross-reference critical values
        if action.action_type == ActionType.APPROVE_PLAN:
            # L3 can cross-check reported vs true values with probability
            if action.is_erroneous and random.random() < self.detection_probability:
                action.detected = True
                action.detection_timestamp = action.timestamp
                self.detected_errors.append(action.action_id)
                return False, "Blocked: anomaly detected in treatment approval"
        
        # Check 3: Anomaly detection on medication orders
        if action.action_type == ActionType.ORDER_MEDICATION:
            if action.is_erroneous and random.random() < self.detection_probability:
                action.detected = True
                action.detection_timestamp = action.timestamp
                self.detected_errors.append(action.action_id)
                return False, "Blocked: anomaly detected in medication order"
        
        # Check 4: Final safety check before execution
        if action.action_type == ActionType.EXECUTE_TREATMENT:
            # L3: Cross-reference entire chain before execution
            if action.is_erroneous and random.random() < self.detection_probability * 1.2:
                action.detected = True
                action.detection_timestamp = action.timestamp
                self.detected_errors.append(action.action_id)
                return False, "Blocked: safety check failed before execution"
        
        return True, None
    
    def analyze_cascades(self) -> List[CascadeEvent]:
        """Analyze cascades with detection information"""
        error_origins = [a for a in self.actions 
                        if a.is_erroneous and not any(
                            self.action_registry.get(b, ClinicalAction("","",AgentRole.LAB_PROCESSING,
                                ActionType.READ_LAB,"",0,{},{})).is_erroneous 
                            for b in a.based_on if b in self.action_registry
                        )]
        
        cascades = []
        for origin in error_origins:
            affected = self._trace_downstream(origin.action_id)
            patients = set(a.patient_id for a in affected + [origin])
            
            # Check if any action in cascade was detected
            all_in_cascade = [origin] + affected
            detected_in_cascade = [a for a in all_in_cascade if a.detected]
            
            if detected_in_cascade:
                detection_depth = min(all_in_cascade.index(a) for a in detected_in_cascade)
            else:
                detection_depth = None
            
            # Harm only occurs if error reaches execution AND not detected
            harm = any(
                a.output_data.get("harm_occurred", False) and not a.detected
                for a in affected
            )
            
            cascades.append(CascadeEvent(
                origin_action_id=origin.action_id,
                affected_action_ids=[a.action_id for a in affected],
                depth=len(affected),
                patients_affected=patients,
                detected=bool(detected_in_cascade),
                detection_depth=detection_depth,
                harm_occurred=harm
            ))
        
        self.cascades = cascades
        return cascades
    
    def _trace_downstream(self, action_id: str) -> List[ClinicalAction]:
        """Trace downstream, stopping at blocked actions"""
        affected = []
        for action in self.actions:
            if action_id in action.based_on:
                if action.action_id not in self.blocked_actions:
                    affected.append(action)
                    affected.extend(self._trace_downstream(action.action_id))
        return affected


# =============================================================================
# Simulation
# =============================================================================

def create_patient_cohort(n_patients: int) -> List[Patient]:
    """Create patients with varying eligibility"""
    patients = []
    for i in range(n_patients):
        true_platelets = random.choices(
            [random.randint(40, 80), random.randint(100, 200)],
            weights=[0.3, 0.7]  # 30% have low platelets
        )[0]
        
        patients.append(Patient(
            id=f"PT-{i:03d}",
            true_platelet_count=true_platelets,
            true_glucose=random.randint(100, 250),
            has_diabetes=random.random() < 0.5,
            chemo_eligible=true_platelets >= 100
        ))
    return patients


def run_simulation(patients: List[Patient], level: str, 
                   error_rate: float = 0.08) -> Dict:
    """
    Run cascade simulation at given coordination level.
    
    Pipeline: Lab → Oncology → Pharmacy → Scheduling → Nursing
    Each step can introduce errors; each step can propagate upstream errors.
    """
    
    if level == "L1":
        env = L1Environment()
    else:
        env = L3Environment(detection_probability=0.75)
    
    # Create agents with specified error rate
    lab_agent = LabProcessingAgent("LAB-001", error_rate=error_rate)
    onco_agent = OncologyAgent("ONCO-001", error_rate=error_rate * 0.6)
    pharm_agent = PharmacyAgent("PHARM-001", error_rate=error_rate * 0.4)
    sched_agent = SchedulingAgent("SCHED-001", error_rate=error_rate * 0.2)
    nurse_agent = NursingAgent("NURSE-001", error_rate=error_rate * 0.1)
    
    timestamp = 0
    
    for patient in patients:
        # Pipeline execution
        lab_action = lab_agent.process_labs(patient, timestamp)
        env.record_action(lab_action)
        
        if level == "L3":
            passed, msg = env.check_action(lab_action, patient)
            if not passed:
                timestamp += 1
                continue
        
        onco_action = onco_agent.evaluate_chemo_eligibility(patient, lab_action, timestamp)
        env.record_action(onco_action)
        
        if level == "L3":
            passed, msg = env.check_action(onco_action, patient)
            if not passed:
                timestamp += 1
                continue
        
        pharm_action = pharm_agent.process_order(patient, onco_action, timestamp)
        env.record_action(pharm_action)
        
        if level == "L3":
            passed, msg = env.check_action(pharm_action, patient)
            if not passed:
                timestamp += 1
                continue
        
        sched_action = sched_agent.schedule_treatment(patient, pharm_action, timestamp)
        env.record_action(sched_action)
        
        if level == "L3":
            passed, msg = env.check_action(sched_action, patient)
            if not passed:
                timestamp += 1
                continue
        
        nurse_action = nurse_agent.execute_treatment(patient, sched_action, timestamp)
        env.record_action(nurse_action)
        
        if level == "L3":
            passed, msg = env.check_action(nurse_action, patient)
        
        timestamp += 1
    
    # Analyze cascades
    cascades = env.analyze_cascades()
    
    # Calculate metrics
    total_errors_introduced = sum(1 for a in env.actions if a.is_erroneous)
    total_cascades = len(cascades)
    total_harm = sum(1 for c in cascades if c.harm_occurred)
    detected_cascades = sum(1 for c in cascades if c.detected)
    avg_cascade_depth = np.mean([c.depth for c in cascades]) if cascades else 0
    
    # For L3, count blocked actions
    blocked = len(env.blocked_actions) if level == "L3" else 0
    
    return {
        "level": level,
        "n_patients": len(patients),
        "total_actions": len(env.actions),
        "errors_introduced": total_errors_introduced,
        "cascades": total_cascades,
        "harm_events": total_harm,
        "detected_cascades": detected_cascades,
        "blocked_actions": blocked,
        "avg_cascade_depth": avg_cascade_depth,
        "harm_rate": total_harm / len(patients) if patients else 0,
        "detection_rate": detected_cascades / total_cascades if total_cascades > 0 else 0,
        "cascade_details": cascades
    }


def run_comparative_analysis(n_runs: int = 50, n_patients: int = 100) -> Dict:
    """Run multiple simulations comparing L1 vs L3"""
    
    results = {"L1": defaultdict(list), "L3": defaultdict(list)}
    
    for run in range(n_runs):
        patients = create_patient_cohort(n_patients)
        
        for level in ["L1", "L3"]:
            r = run_simulation(patients.copy(), level, error_rate=0.10)
            
            results[level]["cascades"].append(r["cascades"])
            results[level]["harm_events"].append(r["harm_events"])
            results[level]["detected"].append(r["detected_cascades"])
            results[level]["blocked"].append(r["blocked_actions"])
            results[level]["avg_depth"].append(r["avg_cascade_depth"])
            results[level]["harm_rate"].append(r["harm_rate"])
    
    return results


def create_figure(results: Dict):
    """Create publication figure for cascade failure analysis"""
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    
    L1_COLOR = '#E74C3C'
    L3_COLOR = '#27AE60'
    DETECTED_COLOR = '#3498DB'
    
    # Panel A: Harm Events (the key safety metric)
    ax = axes[0]
    
    l1_harm = results["L1"]["harm_events"]
    l3_harm = results["L3"]["harm_events"]
    
    positions = [0, 1]
    bp = ax.boxplot([l1_harm, l3_harm], positions=positions, widths=0.5,
                    patch_artist=True)
    
    bp['boxes'][0].set_facecolor(L1_COLOR)
    bp['boxes'][1].set_facecolor(L3_COLOR)
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['L1\n(No Coordination)', 'L3\n(CIRC Protocol)'])
    ax.set_ylabel('Harm Events per 100 Patients')
    ax.set_title('A. Patient Harm from\nCascade Failures', fontweight='bold')
    
    # Add reduction annotation
    l1_mean = np.mean(l1_harm)
    l3_mean = np.mean(l3_harm)
    reduction = (l1_mean - l3_mean) / l1_mean * 100 if l1_mean > 0 else 0
    
    ax.annotate(f'{reduction:.0f}% harm\nreduction', 
               xy=(0.5, (l1_mean + l3_mean) / 2),
               fontsize=11, ha='center', fontweight='bold', color='#27AE60')
    
    # Panel B: Error Detection and Containment
    ax = axes[1]
    
    l3_cascades = np.mean(results["L3"]["cascades"])
    l3_detected = np.mean(results["L3"]["detected"])
    l3_undetected = l3_cascades - l3_detected
    l1_cascades = np.mean(results["L1"]["cascades"])
    
    x = np.arange(2)
    width = 0.6
    
    # L1: All cascades undetected
    ax.bar(0, l1_cascades, width, color=L1_COLOR, alpha=0.7, label='Undetected')
    
    # L3: Split into detected vs undetected
    ax.bar(1, l3_detected, width, color=DETECTED_COLOR, alpha=0.7, label='Detected & Blocked')
    ax.bar(1, l3_undetected, width, bottom=l3_detected, color=L1_COLOR, alpha=0.4)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['L1\n(No Coordination)', 'L3\n(CIRC Protocol)'])
    ax.set_ylabel('Error Cascades per 100 Patients')
    ax.set_title('B. Error Detection\nand Containment', fontweight='bold')
    ax.legend(loc='upper right')
    
    # Detection rate annotation
    detection_rate = l3_detected / l3_cascades * 100 if l3_cascades > 0 else 0
    ax.annotate(f'{detection_rate:.0f}% detected\nbefore harm', 
               xy=(1, l3_detected / 2), 
               fontsize=10, ha='center', color='white', fontweight='bold')
    
    # Panel C: Cascade Propagation Depth
    ax = axes[2]
    
    l1_depth = results["L1"]["avg_depth"]
    l3_depth = results["L3"]["avg_depth"]
    
    bp = ax.boxplot([l1_depth, l3_depth], positions=[0, 1], widths=0.5,
                    patch_artist=True)
    
    bp['boxes'][0].set_facecolor(L1_COLOR)
    bp['boxes'][1].set_facecolor(L3_COLOR)
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['L1\n(No Coordination)', 'L3\n(CIRC Protocol)'])
    ax.set_ylabel('Average Cascade Depth\n(agents affected)')
    ax.set_title('C. Error Propagation\nContainment', fontweight='bold')
    
    # Add annotations
    l1_depth_mean = np.mean(l1_depth)
    l3_depth_mean = np.mean(l3_depth)
    
    ax.annotate(f'Full propagation\n({l1_depth_mean:.1f} agents)', 
               xy=(0, l1_depth_mean), xytext=(-0.3, l1_depth_mean + 0.3),
               fontsize=9, ha='center', color=L1_COLOR)
    ax.annotate(f'Early containment\n({l3_depth_mean:.1f} agents)', 
               xy=(1, l3_depth_mean), xytext=(1.3, l3_depth_mean + 0.3),
               fontsize=9, ha='center', color=L3_COLOR)
    
    plt.tight_layout()
    plt.savefig('circ_cascade_demo.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('circ_cascade_demo.pdf', bbox_inches='tight')
    
    return fig


def create_propagation_diagram():
    """Create a diagram showing cascade propagation with/without CIRC"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Define pipeline stages
    stages = ['Lab\nProcessing', 'Oncology\nReview', 'Pharmacy\nOrder', 'Scheduling', 'Nursing\nExecution']
    n_stages = len(stages)
    
    # L1: Error propagates unchecked
    ax = axes[0]
    ax.set_xlim(-0.5, n_stages - 0.5)
    ax.set_ylim(-0.5, 1.5)
    
    # Draw stages
    for i, stage in enumerate(stages):
        color = '#E74C3C' if i > 0 else '#E74C3C'  # All red after error
        ax.add_patch(plt.Circle((i, 0.5), 0.3, color=color, alpha=0.7))
        ax.text(i, -0.1, stage, ha='center', va='top', fontsize=9)
        
        # Draw arrows
        if i < n_stages - 1:
            ax.annotate('', xy=(i + 0.65, 0.5), xytext=(i + 0.35, 0.5),
                       arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
    
    # Error origin marker
    ax.annotate('Error\nintroduced', xy=(0, 0.9), xytext=(0, 1.3),
               fontsize=9, ha='center', color='#C0392B',
               arrowprops=dict(arrowstyle='->', color='#C0392B'))
    
    # Harm marker
    ax.annotate('HARM', xy=(4, 0.5), xytext=(4.5, 1.0),
               fontsize=11, ha='center', color='#C0392B', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#C0392B'))
    
    ax.set_title('L1: Error Propagates Unchecked', fontweight='bold', fontsize=12)
    ax.axis('off')
    
    # L3: Error detected and contained
    ax = axes[1]
    ax.set_xlim(-0.5, n_stages - 0.5)
    ax.set_ylim(-0.5, 1.5)
    
    # Draw stages
    for i, stage in enumerate(stages):
        if i == 0:
            color = '#E74C3C'  # Error origin
        elif i == 1:
            color = '#3498DB'  # Detection point
        else:
            color = '#27AE60'  # Protected
        ax.add_patch(plt.Circle((i, 0.5), 0.3, color=color, alpha=0.7))
        ax.text(i, -0.1, stage, ha='center', va='top', fontsize=9)
        
        # Draw arrows
        if i < n_stages - 1:
            if i == 0:
                # Blocked arrow
                ax.annotate('', xy=(i + 0.5, 0.5), xytext=(i + 0.35, 0.5),
                           arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
                ax.plot([i + 0.55, i + 0.65], [0.4, 0.6], 'r-', lw=3)
                ax.plot([i + 0.55, i + 0.65], [0.6, 0.4], 'r-', lw=3)
            else:
                ax.annotate('', xy=(i + 0.65, 0.5), xytext=(i + 0.35, 0.5),
                           arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2))
    
    # Error origin marker
    ax.annotate('Error\nintroduced', xy=(0, 0.9), xytext=(0, 1.3),
               fontsize=9, ha='center', color='#C0392B',
               arrowprops=dict(arrowstyle='->', color='#C0392B'))
    
    # Detection marker
    ax.annotate('CIRC detects\n& blocks', xy=(1, 0.9), xytext=(1, 1.3),
               fontsize=9, ha='center', color='#2980B9', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#2980B9'))
    
    # No harm marker
    ax.annotate('No harm', xy=(4, 0.5), xytext=(4.5, 1.0),
               fontsize=11, ha='center', color='#27AE60', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#27AE60'))
    
    ax.set_title('L3: Error Detected and Contained', fontweight='bold', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('circ_cascade_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('circ_cascade_diagram.pdf', bbox_inches='tight')
    
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CIRC Cascade Failure Demonstration")
    print("Error Propagation and Containment Analysis")
    print("=" * 70)
    print()
    
    print("Scenario: Clinical pipeline (Lab → Oncology → Pharmacy → Scheduling → Nursing)")
    print("  - Errors introduced at each stage with 10% base rate")
    print("  - Errors propagate downstream, compounding")
    print("  - Harm occurs when erroneous treatment reaches execution")
    print()
    
    print("Running comparative analysis (50 runs, 100 patients each)...")
    results = run_comparative_analysis(n_runs=50, n_patients=100)
    
    print()
    print("RESULTS")
    print("-" * 70)
    print(f"{'Metric':<35} {'L1 (Isolated)':<18} {'L3 (CIRC)':<18}")
    print("-" * 70)
    
    l1_harm = np.mean(results["L1"]["harm_events"])
    l3_harm = np.mean(results["L3"]["harm_events"])
    print(f"{'Harm Events (per 100 patients)':<35} {l1_harm:<18.1f} {l3_harm:<18.1f}")
    
    l1_cascades = np.mean(results["L1"]["cascades"])
    l3_cascades = np.mean(results["L3"]["cascades"])
    print(f"{'Error Cascades':<35} {l1_cascades:<18.1f} {l3_cascades:<18.1f}")
    
    l3_detected = np.mean(results["L3"]["detected"])
    print(f"{'Cascades Detected':<35} {'0':<18} {l3_detected:<18.1f}")
    
    l3_blocked = np.mean(results["L3"]["blocked"])
    print(f"{'Actions Blocked':<35} {'0':<18} {l3_blocked:<18.1f}")
    
    l1_depth = np.mean(results["L1"]["avg_depth"])
    l3_depth = np.mean(results["L3"]["avg_depth"])
    print(f"{'Avg Cascade Depth (agents)':<35} {l1_depth:<18.1f} {l3_depth:<18.1f}")
    
    print("-" * 70)
    print()
    
    # Key findings
    harm_reduction = (l1_harm - l3_harm) / l1_harm * 100 if l1_harm > 0 else 0
    detection_rate = l3_detected / l3_cascades * 100 if l3_cascades > 0 else 0
    depth_reduction = (l1_depth - l3_depth) / l1_depth * 100 if l1_depth > 0 else 0
    
    print("KEY FINDINGS:")
    print(f"  1. Harm reduction: {harm_reduction:.0f}% fewer adverse events with CIRC")
    print(f"  2. Detection rate: {detection_rate:.0f}% of cascades caught before harm")
    print(f"  3. Containment: {depth_reduction:.0f}% reduction in cascade propagation depth")
    print()
    print("  Clinical interpretation:")
    print("  - L1: Errors invisible to other agents; cascade until patient harm")
    print("  - L3: Coordination infrastructure provides checkpoints that catch errors")
    print("  - This is the 'Do No Harm' implementation: error containment infrastructure")
    print()
    
    print("Generating figures...")
    create_figure(results)
    create_propagation_diagram()
    print("Figures saved as circ_cascade_demo.png/.pdf and circ_cascade_diagram.png/.pdf")
    print()
    
    # Export JSON
    output = {
        "scenario": "Cascade Failure Propagation and Containment",
        "paper": "Asimov's Laws for Clinical AI Agents (Xu, Chopra, Ryu 2025)",
        "connection_to_asimov": "L0: Do no harm - implemented via error containment infrastructure",
        "parameters": {
            "n_runs": 50,
            "n_patients": 100,
            "error_rate": 0.10,
            "l3_detection_probability": 0.75,
            "pipeline_stages": ["Lab", "Oncology", "Pharmacy", "Scheduling", "Nursing"]
        },
        "results": {
            "L1": {
                "harm_events_mean": l1_harm,
                "harm_events_std": np.std(results["L1"]["harm_events"]),
                "cascades_mean": l1_cascades,
                "avg_depth_mean": l1_depth,
                "detection_rate": 0
            },
            "L3": {
                "harm_events_mean": l3_harm,
                "harm_events_std": np.std(results["L3"]["harm_events"]),
                "cascades_mean": l3_cascades,
                "detected_mean": l3_detected,
                "blocked_mean": l3_blocked,
                "avg_depth_mean": l3_depth,
                "detection_rate": detection_rate / 100
            }
        },
        "key_findings": {
            "harm_reduction_pct": harm_reduction,
            "detection_rate_pct": detection_rate,
            "depth_reduction_pct": depth_reduction
        },
        "clinical_interpretation": {
            "L1_failure_mode": "Errors propagate invisibly through pipeline until harm manifests",
            "L3_protection": "Cross-reference checks and anomaly detection catch errors before harm",
            "asimov_connection": "CIRC provides the infrastructure for 'Do No Harm' to be operationalized"
        }
    }
    
    with open("circ_cascade_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("Results exported to circ_cascade_results.json")
    print("=" * 70)
