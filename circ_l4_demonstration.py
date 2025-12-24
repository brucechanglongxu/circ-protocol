"""
CIRC-L4 Demonstration: Multi-Agent Signal Aggregation

This simulation demonstrates L4's unique capability: detecting emergent patterns
from aggregating weak signals across multiple agents that no single agent can see.

Scenario: 58-year-old woman with HER2+ breast cancer on TCHP + diabetes
- Declining platelets (oncology agent sees this)
- Vague dizziness, worsening headache (triage agent parses these)
- Unexplained glucose elevation (diabetes agent monitors this)
- Nocturnal heart rate variability (wearable data routed to diabetes agent)

Under L1-L3: Each signal is below individual agent's alert threshold.
Under L4: Aggregate "concern signal" crosses population-level detection threshold.

This maps to the paper's key claim:
"The same infrastructure that caught three other hemorrhages presenting with classic 
sudden-onset symptoms failed on this subacute presentation—not because any agent 
behaved incorrectly, but because the infrastructure provides no mechanism for agents 
to share context or negotiate priorities."
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
from collections import defaultdict
import json

# Reproducibility
random.seed(42)
np.random.seed(42)

# =============================================================================
# Data Structures
# =============================================================================

class SignalType(Enum):
    PLATELET_DECLINE = "platelet_decline"
    HEADACHE = "headache"
    DIZZINESS = "dizziness"
    GLUCOSE_ELEVATION = "glucose_elevation"
    HEART_RATE_VARIABILITY = "hrv_anomaly"
    FATIGUE = "fatigue"
    NAUSEA = "nausea"

class Urgency(Enum):
    ROUTINE = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    CRITICAL = 5

@dataclass
class ClinicalSignal:
    """A signal observed by an agent"""
    signal_type: SignalType
    patient_id: str
    agent_id: str
    timestamp: int  # Day
    raw_value: float  # Normalized 0-1
    agent_interpretation: Urgency
    confidence: float  # Agent's confidence in interpretation

@dataclass
class Patient:
    """Patient with evolving clinical state"""
    id: str
    has_hemorrhage: bool = False  # Ground truth
    presentation_type: str = "none"  # "classic" or "subacute"
    
@dataclass
class AggregateSignal:
    """L4 aggregate signal across agents"""
    patient_id: str
    timestamp: int
    contributing_signals: List[ClinicalSignal]
    aggregate_concern: float  # 0-1
    triggered_escalation: bool

# =============================================================================
# Agent Definitions
# =============================================================================

class BaseAgent:
    """Base agent with signal emission"""
    
    def __init__(self, agent_id: str, domain: str):
        self.id = agent_id
        self.domain = domain
        self.alert_threshold = 0.7  # Individual agent threshold
        
    def interpret_signal(self, raw_value: float, signal_type: SignalType) -> tuple:
        """Convert raw signal to urgency + confidence"""
        # Each agent has domain-specific thresholds
        if raw_value > 0.8:
            return Urgency.HIGH, 0.9
        elif raw_value > 0.6:
            return Urgency.MODERATE, 0.7
        elif raw_value > 0.4:
            return Urgency.LOW, 0.6
        else:
            return Urgency.ROUTINE, 0.8


class OncologyAgent(BaseAgent):
    """Monitors labs, treatment response"""
    
    def __init__(self):
        super().__init__("ONCO-001", "oncology")
        self.platelet_history = defaultdict(list)
        
    def observe_platelets(self, patient_id: str, value: float, day: int) -> Optional[ClinicalSignal]:
        """
        Oncology agent sees platelet trends.
        In the paper: "Sees declining platelets (95k → 82k → 68k)"
        
        Returns concern signal: higher when platelets lower and declining
        """
        self.platelet_history[patient_id].append(value)
        
        # Absolute level concern (normalized to 0-1)
        # 150k = 0, 50k = 0.8
        level_concern = max(0, min(1, (130 - value) / 100))
        
        # Trend concern
        history = self.platelet_history[patient_id]
        if len(history) >= 2:
            daily_drop = history[-2] - history[-1]
            trend_concern = max(0, min(0.3, daily_drop / 30))  # Up to 0.3 for rapid decline
        else:
            trend_concern = 0
            
        raw_concern = min(1.0, level_concern + trend_concern)
        
        urgency, confidence = self.interpret_signal(raw_concern, SignalType.PLATELET_DECLINE)
        
        return ClinicalSignal(
            signal_type=SignalType.PLATELET_DECLINE,
            patient_id=patient_id,
            agent_id=self.id,
            timestamp=day,
            raw_value=raw_concern,
            agent_interpretation=urgency,
            confidence=confidence
        )


class TriageAgent(BaseAgent):
    """Parses symptoms from patient messages"""
    
    def __init__(self):
        super().__init__("TRIAGE-001", "triage")
        
    def parse_symptom(self, patient_id: str, symptom: str, severity: float, day: int) -> ClinicalSignal:
        """
        Triage agent parses symptoms.
        In the paper: "vague dizziness", "worsening headache" parsed as low priority
        
        Key: Vague symptoms are discounted but still contribute to aggregate
        """
        symptom_map = {
            "headache": SignalType.HEADACHE,
            "dizziness": SignalType.DIZZINESS,
            "fatigue": SignalType.FATIGUE,
            "nausea": SignalType.NAUSEA
        }
        signal_type = symptom_map.get(symptom, SignalType.FATIGUE)
        
        # Vague symptoms discounted but not eliminated
        # This is the key - each signal stays below 0.7 but contributes to aggregate
        adjusted_severity = severity * 0.85
        
        urgency, confidence = self.interpret_signal(adjusted_severity, signal_type)
        
        return ClinicalSignal(
            signal_type=signal_type,
            patient_id=patient_id,
            agent_id=self.id,
            timestamp=day,
            raw_value=adjusted_severity,
            agent_interpretation=urgency,
            confidence=confidence
        )


class DiabetesAgent(BaseAgent):
    """Monitors glucose, wearable data"""
    
    def __init__(self):
        super().__init__("DIABETES-001", "diabetes")
        
    def observe_glucose(self, patient_id: str, value: float, day: int) -> ClinicalSignal:
        """Monitor glucose - unexplained elevation is concerning"""
        # Normalized concern: 0 at 140, rises to 0.6 at 250
        raw_concern = max(0, min(0.7, (value - 140) / 170))
        
        urgency, confidence = self.interpret_signal(raw_concern, SignalType.GLUCOSE_ELEVATION)
        
        return ClinicalSignal(
            signal_type=SignalType.GLUCOSE_ELEVATION,
            patient_id=patient_id,
            agent_id=self.id,
            timestamp=day,
            raw_value=raw_concern,
            agent_interpretation=urgency,
            confidence=confidence
        )
    
    def observe_hrv(self, patient_id: str, anomaly_score: float, day: int) -> ClinicalSignal:
        """
        Wearable data - diabetes agent receives but discounts cardiac significance.
        From paper: "her wearable data showing nocturnal heart rate changes reaches 
        only the diabetes agent, which classifies the pattern as noise"
        
        Key: Signal passes through but at reduced weight
        """
        # Diabetes agent partially discounts HRV (outside domain)
        adjusted_score = anomaly_score * 0.75
        
        urgency, confidence = self.interpret_signal(adjusted_score, SignalType.HEART_RATE_VARIABILITY)
        
        return ClinicalSignal(
            signal_type=SignalType.HEART_RATE_VARIABILITY,
            patient_id=patient_id,
            agent_id=self.id,
            timestamp=day,
            raw_value=adjusted_score,
            agent_interpretation=urgency,
            confidence=confidence * 0.7  # Low confidence - outside domain
        )


# =============================================================================
# Coordination Levels
# =============================================================================

class L3Coordinator:
    """
    L3: Agents coordinate on specific actions but don't aggregate signals.
    Each agent makes independent escalation decisions.
    """
    
    def __init__(self, escalation_threshold: float = 0.7):
        self.escalation_threshold = escalation_threshold
        self.escalations = []
        
    def process_signals(self, signals: List[ClinicalSignal], day: int) -> Dict:
        """
        L3: Each agent independently decides if its signal warrants escalation.
        No cross-agent signal aggregation.
        """
        escalated = False
        escalation_reason = None
        
        for signal in signals:
            # Each agent uses its own threshold
            if signal.raw_value >= self.escalation_threshold:
                escalated = True
                escalation_reason = f"{signal.agent_id}: {signal.signal_type.value}"
                self.escalations.append({
                    "day": day,
                    "patient": signal.patient_id,
                    "reason": escalation_reason,
                    "signal_value": signal.raw_value
                })
                break  # First escalation wins
        
        return {
            "escalated": escalated,
            "reason": escalation_reason,
            "aggregate_concern": max(s.raw_value for s in signals) if signals else 0,
            "method": "individual_threshold"
        }


class L4Coordinator:
    """
    L4: System-level signal aggregation across agents.
    
    Key capability: Detects emergent patterns from combining weak signals
    that no single agent would escalate on.
    
    From paper: "aggregate resource and priority signals" and 
    "privacy-preserving summaries of system-level state"
    """
    
    def __init__(self, 
                 individual_threshold: float = 0.7,
                 aggregate_threshold: float = 0.45,  # Lower than individual
                 convergence_bonus: float = 0.25):   # Significant bonus for multi-domain
        self.individual_threshold = individual_threshold
        self.aggregate_threshold = aggregate_threshold
        self.convergence_bonus = convergence_bonus
        self.escalations = []
        self.aggregate_signals = []
        
    def compute_aggregate_concern(self, signals: List[ClinicalSignal]) -> float:
        """
        L4 primitive: Aggregate concern signal.
        
        Key insight: Multiple weak signals from different domains
        should increase concern more than a single moderate signal.
        
        This is the "multi-agent concern convergence" from the paper.
        """
        if not signals:
            return 0
            
        # Get raw values
        values = [s.raw_value for s in signals]
        
        # Base: Use max of top signals (not just average)
        sorted_values = sorted(values, reverse=True)
        if len(sorted_values) >= 3:
            base = (sorted_values[0] + sorted_values[1] + sorted_values[2]) / 3
        else:
            base = np.mean(sorted_values)
        
        # L4 KEY FEATURE: Cross-domain convergence bonus
        # Multiple independent domains raising concern is MORE significant
        # than a single domain with higher concern
        concerned_domains = sum(1 for s in signals if s.raw_value > 0.35)
        
        # Convergence multiplier increases with number of concerned domains
        if concerned_domains >= 4:
            convergence_multiplier = 1.0 + self.convergence_bonus * 2.5
        elif concerned_domains >= 3:
            convergence_multiplier = 1.0 + self.convergence_bonus * 1.5
        elif concerned_domains >= 2:
            convergence_multiplier = 1.0 + self.convergence_bonus
        else:
            convergence_multiplier = 1.0
            
        aggregate = min(1.0, base * convergence_multiplier)
        
        return aggregate
    
    def process_signals(self, signals: List[ClinicalSignal], day: int) -> Dict:
        """
        L4: Aggregate signals across agents, apply convergence detection.
        """
        # First check individual thresholds (L3 capability)
        individual_escalation = any(s.raw_value >= self.individual_threshold for s in signals)
        
        # L4: Compute aggregate concern
        aggregate_concern = self.compute_aggregate_concern(signals)
        
        # L4 escalation: either individual threshold OR aggregate threshold
        aggregate_escalation = aggregate_concern >= self.aggregate_threshold
        
        escalated = individual_escalation or aggregate_escalation
        
        if escalated:
            if individual_escalation:
                reason = "individual_threshold"
                trigger = max(signals, key=lambda s: s.raw_value)
                reason_detail = f"{trigger.agent_id}: {trigger.signal_type.value}"
            else:
                reason = "aggregate_convergence"
                contributing = [s for s in signals if s.raw_value > 0.3]
                reason_detail = f"Multi-domain concern: {[s.agent_id for s in contributing]}"
                
            self.escalations.append({
                "day": day,
                "patient": signals[0].patient_id if signals else "unknown",
                "reason": reason,
                "detail": reason_detail,
                "aggregate_concern": aggregate_concern
            })
        
        # Record aggregate signal
        if signals:
            self.aggregate_signals.append(AggregateSignal(
                patient_id=signals[0].patient_id,
                timestamp=day,
                contributing_signals=signals,
                aggregate_concern=aggregate_concern,
                triggered_escalation=escalated
            ))
        
        return {
            "escalated": escalated,
            "reason": reason if escalated else None,
            "aggregate_concern": aggregate_concern,
            "method": "aggregate_convergence" if (escalated and not individual_escalation) else "individual_threshold"
        }


# =============================================================================
# Patient Trajectory Simulation
# =============================================================================

def simulate_subacute_hemorrhage_trajectory(patient_id: str, n_days: int = 7) -> Dict:
    """
    Simulate the 58-year-old's trajectory from the paper.
    
    CRITICAL DESIGN:
    - Each individual signal peaks at ~0.45-0.55 (WELL BELOW 0.7 L3 threshold)
    - 4 signals at ~0.45 with L4 convergence bonus → aggregate crosses 0.45 L4 threshold
    
    This creates the scenario where L3 (individual thresholds) consistently fails
    but L4 (aggregate + convergence) consistently succeeds.
    """
    trajectory = {
        "patient_id": patient_id,
        "ground_truth": "subacute_subdural_hematoma",
        "days": []
    }
    
    for day in range(n_days):
        # Platelet trajectory: 95k → ~75k (creates concern ~0.45-0.55, well below 0.7)
        platelet_base = 95 - (day * 3) + random.gauss(0, 1.5)
        platelet_value = max(73, platelet_base)
        
        # Symptoms: raw values that after processing stay at ~0.40-0.50
        headache_severity = min(0.52, 0.18 + day * 0.05 + random.gauss(0, 0.02))
        dizziness_severity = min(0.45, 0.10 + day * 0.05 + random.gauss(0, 0.02))
        
        # Glucose: peaks at ~0.45 concern
        glucose = 150 + day * 9 + random.gauss(0, 5)  # ~150 to ~210
        
        # HRV: peaks at ~0.50 raw
        hrv_anomaly = min(0.50, 0.18 + day * 0.05 + random.gauss(0, 0.02))
        
        trajectory["days"].append({
            "day": day,
            "platelets": platelet_value,
            "headache": headache_severity,
            "dizziness": dizziness_severity,
            "glucose": glucose,
            "hrv_anomaly": hrv_anomaly
        })
    
    return trajectory


def simulate_classic_hemorrhage_trajectory(patient_id: str, n_days: int = 7) -> Dict:
    """
    Classic presentation: sudden onset, clearly above threshold.
    Both L3 and L4 should catch this.
    """
    trajectory = {
        "patient_id": patient_id,
        "ground_truth": "classic_hemorrhage",
        "days": []
    }
    
    for day in range(n_days):
        if day < 5:
            # Normal before sudden onset
            platelet_value = 120 + random.gauss(0, 5)
            headache_severity = 0.1 + random.gauss(0, 0.05)
            dizziness_severity = 0.1 + random.gauss(0, 0.05)
            glucose = 130 + random.gauss(0, 10)
            hrv_anomaly = 0.1 + random.gauss(0, 0.05)
        else:
            # Day 5+: Sudden classic presentation
            platelet_value = 45 + random.gauss(0, 5)  # Precipitous drop
            headache_severity = 0.95  # "Thunderclap" headache
            dizziness_severity = 0.8
            glucose = 220 + random.gauss(0, 10)
            hrv_anomaly = 0.85
        
        trajectory["days"].append({
            "day": day,
            "platelets": max(30, platelet_value),
            "headache": min(1.0, max(0, headache_severity)),
            "dizziness": min(1.0, max(0, dizziness_severity)),
            "glucose": glucose,
            "hrv_anomaly": min(1.0, max(0, hrv_anomaly))
        })
    
    return trajectory


def simulate_healthy_trajectory(patient_id: str, n_days: int = 7) -> Dict:
    """Normal patient - no hemorrhage"""
    trajectory = {
        "patient_id": patient_id,
        "ground_truth": "healthy",
        "days": []
    }
    
    for day in range(n_days):
        trajectory["days"].append({
            "day": day,
            "platelets": 150 + random.gauss(0, 10),
            "headache": max(0, 0.1 + random.gauss(0, 0.1)),
            "dizziness": max(0, 0.05 + random.gauss(0, 0.05)),
            "glucose": 120 + random.gauss(0, 15),
            "hrv_anomaly": max(0, 0.1 + random.gauss(0, 0.1))
        })
    
    return trajectory


# =============================================================================
# Run Simulation
# =============================================================================

def run_simulation(n_subacute: int = 10, n_classic: int = 5, n_healthy: int = 85) -> Dict:
    """
    Run simulation comparing L3 vs L4 detection.
    
    Population:
    - n_subacute: Subacute hemorrhages (the hard cases - weak converging signals)
    - n_classic: Classic hemorrhages (easy cases - strong individual signals)
    - n_healthy: Healthy patients (should not escalate)
    """
    # Initialize agents
    oncology = OncologyAgent()
    triage = TriageAgent()
    diabetes = DiabetesAgent()
    
    # Initialize coordinators
    # L3: Only individual thresholds (0.7)
    l3 = L3Coordinator(escalation_threshold=0.7)
    
    # L4: Individual threshold (0.7) + aggregate threshold (0.45) with convergence bonus
    l4 = L4Coordinator(individual_threshold=0.7, aggregate_threshold=0.45, convergence_bonus=0.25)
    
    # Generate patient trajectories
    patients = []
    
    for i in range(n_subacute):
        patients.append(simulate_subacute_hemorrhage_trajectory(f"SUBACUTE-{i:03d}"))
    
    for i in range(n_classic):
        patients.append(simulate_classic_hemorrhage_trajectory(f"CLASSIC-{i:03d}"))
    
    for i in range(n_healthy):
        patients.append(simulate_healthy_trajectory(f"HEALTHY-{i:03d}"))
    
    # Track outcomes
    results = {
        "L3": {"true_positives": 0, "false_positives": 0, "false_negatives": 0, "true_negatives": 0,
               "detection_day": [], "escalation_details": []},
        "L4": {"true_positives": 0, "false_positives": 0, "false_negatives": 0, "true_negatives": 0,
               "detection_day": [], "escalation_details": []}
    }
    
    for patient in patients:
        patient_id = patient["patient_id"]
        ground_truth = patient["ground_truth"]
        is_hemorrhage = ground_truth in ["subacute_subdural_hematoma", "classic_hemorrhage"]
        
        l3_detected = False
        l4_detected = False
        l3_detection_day = None
        l4_detection_day = None
        
        for day_data in patient["days"]:
            day = day_data["day"]
            
            # Collect signals from all agents
            signals = []
            
            # Oncology observes platelets
            signals.append(oncology.observe_platelets(patient_id, day_data["platelets"], day))
            
            # Triage parses symptoms
            if day_data["headache"] > 0.1:
                signals.append(triage.parse_symptom(patient_id, "headache", day_data["headache"], day))
            if day_data["dizziness"] > 0.1:
                signals.append(triage.parse_symptom(patient_id, "dizziness", day_data["dizziness"], day))
            
            # Diabetes monitors glucose and wearables
            signals.append(diabetes.observe_glucose(patient_id, day_data["glucose"], day))
            signals.append(diabetes.observe_hrv(patient_id, day_data["hrv_anomaly"], day))
            
            # Process through coordinators
            if not l3_detected:
                l3_result = l3.process_signals(signals, day)
                if l3_result["escalated"]:
                    l3_detected = True
                    l3_detection_day = day
                    results["L3"]["escalation_details"].append({
                        "patient": patient_id,
                        "day": day,
                        "ground_truth": ground_truth,
                        "method": l3_result["method"]
                    })
            
            if not l4_detected:
                l4_result = l4.process_signals(signals, day)
                if l4_result["escalated"]:
                    l4_detected = True
                    l4_detection_day = day
                    results["L4"]["escalation_details"].append({
                        "patient": patient_id,
                        "day": day,
                        "ground_truth": ground_truth,
                        "method": l4_result["method"],
                        "aggregate_concern": l4_result["aggregate_concern"]
                    })
        
        # Record outcomes
        for level, detected, detection_day in [("L3", l3_detected, l3_detection_day), 
                                                 ("L4", l4_detected, l4_detection_day)]:
            if is_hemorrhage:
                if detected:
                    results[level]["true_positives"] += 1
                    results[level]["detection_day"].append(detection_day)
                else:
                    results[level]["false_negatives"] += 1
            else:
                if detected:
                    results[level]["false_positives"] += 1
                else:
                    results[level]["true_negatives"] += 1
    
    # Calculate metrics
    for level in ["L3", "L4"]:
        tp = results[level]["true_positives"]
        fp = results[level]["false_positives"]
        fn = results[level]["false_negatives"]
        tn = results[level]["true_negatives"]
        
        results[level]["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        results[level]["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        results[level]["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        results[level]["avg_detection_day"] = np.mean(results[level]["detection_day"]) if results[level]["detection_day"] else None
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def create_l4_figure(results: Dict):
    """Create publication figure for L4 demonstration"""
    
    fig = plt.figure(figsize=(14, 5))
    
    # Color scheme
    L3_COLOR = '#3498DB'  # Blue
    L4_COLOR = '#9B59B6'  # Purple
    
    # Panel A: Detection Performance (Sensitivity)
    ax1 = fig.add_subplot(131)
    
    metrics = ['Sensitivity\n(Hemorrhage Detection)', 'Specificity\n(Avoid False Alarms)']
    l3_values = [results["L3"]["sensitivity"], results["L3"]["specificity"]]
    l4_values = [results["L4"]["sensitivity"], results["L4"]["specificity"]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, l3_values, width, label='L3 (Direct Coordination)', 
                    color=L3_COLOR, alpha=0.8)
    bars2 = ax1.bar(x + width/2, l4_values, width, label='L4 (Aggregate Signals)', 
                    color=L4_COLOR, alpha=0.8)
    
    ax1.set_ylabel('Rate')
    ax1.set_title('A. Detection Performance', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 1.1)
    
    # Add value labels
    for bar, val in zip(bars1, l3_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.0%}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, l4_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.0%}', ha='center', va='bottom', fontsize=10)
    
    # Panel B: Breakdown by presentation type
    ax2 = fig.add_subplot(132)
    
    # Count detections by type
    l3_subacute = sum(1 for e in results["L3"]["escalation_details"] 
                      if "SUBACUTE" in e["patient"])
    l3_classic = sum(1 for e in results["L3"]["escalation_details"] 
                     if "CLASSIC" in e["patient"])
    l4_subacute = sum(1 for e in results["L4"]["escalation_details"] 
                      if "SUBACUTE" in e["patient"])
    l4_classic = sum(1 for e in results["L4"]["escalation_details"] 
                     if "CLASSIC" in e["patient"])
    
    # Assume 10 subacute, 5 classic from simulation
    n_subacute = 10
    n_classic = 5
    
    categories = ['Classic\nPresentation', 'Subacute\nPresentation']
    l3_rates = [l3_classic/n_classic, l3_subacute/n_subacute]
    l4_rates = [l4_classic/n_classic, l4_subacute/n_subacute]
    
    x = np.arange(len(categories))
    
    bars1 = ax2.bar(x - width/2, l3_rates, width, label='L3', color=L3_COLOR, alpha=0.8)
    bars2 = ax2.bar(x + width/2, l4_rates, width, label='L4', color=L4_COLOR, alpha=0.8)
    
    ax2.set_ylabel('Detection Rate')
    ax2.set_title('B. Detection by Presentation Type', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.set_ylim(0, 1.2)
    
    # Highlight the key finding
    improvement = (l4_subacute - l3_subacute) / max(l3_subacute, 1) * 100
    if l4_subacute > l3_subacute:
        ax2.annotate(f'+{l4_subacute - l3_subacute} detected\nby L4 aggregation', 
                    xy=(1 + width/2, l4_rates[1]), xytext=(1.3, l4_rates[1] + 0.15),
                    fontsize=9, ha='left',
                    arrowprops=dict(arrowstyle='->', color='#8E44AD'))
    
    # Add value labels
    for bar, val in zip(bars1, l3_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.0%}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, l4_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.0%}', ha='center', va='bottom', fontsize=10)
    
    # Panel C: Signal trajectory for example patient
    ax3 = fig.add_subplot(133)
    
    # Simulate one subacute patient for visualization
    example = simulate_subacute_hemorrhage_trajectory("EXAMPLE-001")
    days = [d["day"] for d in example["days"]]
    
    # Show actual signal values as they would be processed by agents
    # Platelet concern: higher when lower (normalized)
    platelet_concern = [max(0, min(1, (130 - d["platelets"]) / 100)) for d in example["days"]]
    headache = [d["headache"] * 0.85 for d in example["days"]]  # After triage discount
    glucose_concern = [max(0, min(0.7, (d["glucose"] - 140) / 170)) for d in example["days"]]
    hrv = [d["hrv_anomaly"] * 0.75 for d in example["days"]]  # After diabetes discount
    
    ax3.plot(days, platelet_concern, 'o-', label='Platelet concern', alpha=0.7, markersize=5, linewidth=2)
    ax3.plot(days, headache, 's-', label='Headache (triage)', alpha=0.7, markersize=5, linewidth=2)
    ax3.plot(days, glucose_concern, '^-', label='Glucose concern', alpha=0.7, markersize=5, linewidth=2)
    ax3.plot(days, hrv, 'd-', label='HRV anomaly', alpha=0.7, markersize=5, linewidth=2)
    
    # Individual threshold line
    ax3.axhline(y=0.7, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)
    ax3.text(6.5, 0.72, 'L3 threshold', fontsize=8, color='#E74C3C')
    
    # Compute aggregate concern with convergence
    aggregate = []
    for i in range(len(days)):
        signals = [platelet_concern[i], headache[i], glucose_concern[i], hrv[i]]
        sorted_signals = sorted(signals, reverse=True)
        base = np.mean(sorted_signals[:3]) if len(sorted_signals) >= 3 else np.mean(sorted_signals)
        n_concerned = sum(1 for s in signals if s > 0.35)
        if n_concerned >= 4:
            multiplier = 1.0 + 0.25 * 2.5
        elif n_concerned >= 3:
            multiplier = 1.0 + 0.25 * 1.5
        elif n_concerned >= 2:
            multiplier = 1.0 + 0.25
        else:
            multiplier = 1.0
        aggregate.append(min(1.0, base * multiplier))
    
    ax3.plot(days, aggregate, 'k-', linewidth=3, label='L4 aggregate', alpha=0.9)
    ax3.axhline(y=0.45, color='#8E44AD', linestyle='--', linewidth=2, alpha=0.8)
    ax3.text(6.5, 0.47, 'L4 threshold', fontsize=8, color='#8E44AD')
    
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Concern Signal (0-1)')
    ax3.set_title('C. Subacute Case: Signal Convergence', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=7, ncol=2)
    ax3.set_ylim(0, 0.9)
    ax3.set_xlim(-0.2, 7)
    
    # Mark L4 detection point
    l4_detect_day = next((i for i, v in enumerate(aggregate) if v >= 0.45), None)
    if l4_detect_day is not None:
        ax3.axvline(x=l4_detect_day, color='#8E44AD', linestyle=':', alpha=0.5)
        ax3.scatter([l4_detect_day], [aggregate[l4_detect_day]], s=100, c='#8E44AD', zorder=5, marker='*')
        ax3.annotate(f'L4 detects\nDay {l4_detect_day}', xy=(l4_detect_day, aggregate[l4_detect_day]), 
                    xytext=(l4_detect_day + 0.8, aggregate[l4_detect_day] - 0.12),
                    fontsize=9, color='#8E44AD', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('circ_l4_demo.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('circ_l4_demo.pdf', bbox_inches='tight')
    
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CIRC-L4 Demonstration: Multi-Agent Signal Aggregation")
    print("Subacute Subdural Hematoma Detection")
    print("=" * 70)
    print()
    
    # Run simulation
    print("Running simulation...")
    print("  - 10 subacute hemorrhages (weak signals, below individual thresholds)")
    print("  - 5 classic hemorrhages (strong signals, above individual thresholds)")
    print("  - 85 healthy patients (no hemorrhage)")
    print()
    
    results = run_simulation(n_subacute=10, n_classic=5, n_healthy=85)
    
    # Print results
    print("RESULTS")
    print("-" * 70)
    print(f"{'Metric':<35} {'L3 (Direct)':<18} {'L4 (Aggregate)':<18}")
    print("-" * 70)
    print(f"{'Sensitivity (detect hemorrhages)':<35} {results['L3']['sensitivity']:<18.1%} {results['L4']['sensitivity']:<18.1%}")
    print(f"{'Specificity (avoid false alarms)':<35} {results['L3']['specificity']:<18.1%} {results['L4']['specificity']:<18.1%}")
    print(f"{'True Positives':<35} {results['L3']['true_positives']:<18} {results['L4']['true_positives']:<18}")
    print(f"{'False Negatives (missed)':<35} {results['L3']['false_negatives']:<18} {results['L4']['false_negatives']:<18}")
    print(f"{'False Positives':<35} {results['L3']['false_positives']:<18} {results['L4']['false_positives']:<18}")
    print("-" * 70)
    print()
    
    # Breakdown by type
    l3_subacute = sum(1 for e in results["L3"]["escalation_details"] if "SUBACUTE" in e["patient"])
    l4_subacute = sum(1 for e in results["L4"]["escalation_details"] if "SUBACUTE" in e["patient"])
    l4_by_aggregation = sum(1 for e in results["L4"]["escalation_details"] 
                            if e["method"] == "aggregate_convergence")
    
    print("KEY FINDING: Subacute Hemorrhage Detection")
    print(f"  L3 detected: {l3_subacute}/10 subacute cases")
    print(f"  L4 detected: {l4_subacute}/10 subacute cases")
    print(f"  L4 detections via aggregation (not individual threshold): {l4_by_aggregation}")
    print()
    
    if l4_subacute > l3_subacute:
        print(f"  → L4 caught {l4_subacute - l3_subacute} additional cases that L3 missed")
        print("    These represent the '58-year-old' scenario from the paper:")
        print("    Multiple weak signals converging across domains.")
    print()
    
    # Create figure
    print("Generating publication figure...")
    create_l4_figure(results)
    print("Figure saved as circ_l4_demo.png and .pdf")
    print()
    
    # Export JSON
    output = {
        "scenario": "Subacute Subdural Hematoma Detection",
        "paper": "Asimov's Laws for Clinical AI Agents (Xu, Chopra, Ryu 2025)",
        "l4_capability": "Multi-agent signal aggregation with convergence detection",
        "parameters": {
            "n_subacute": 10,
            "n_classic": 5,
            "n_healthy": 85,
            "l3_threshold": 0.7,
            "l4_individual_threshold": 0.7,
            "l4_aggregate_threshold": 0.45,
            "convergence_bonus": 0.25
        },
        "results": {
            "L3": {
                "sensitivity": results["L3"]["sensitivity"],
                "specificity": results["L3"]["specificity"],
                "subacute_detected": l3_subacute
            },
            "L4": {
                "sensitivity": results["L4"]["sensitivity"],
                "specificity": results["L4"]["specificity"],
                "subacute_detected": l4_subacute,
                "detected_via_aggregation": l4_by_aggregation
            }
        },
        "clinical_interpretation": {
            "L3_limitation": "Each agent applies individual threshold (0.7); weak converging signals missed",
            "L4_advantage": "Aggregate concern signal (threshold 0.45) with convergence bonus detects multi-domain patterns",
            "clinical_analog": "The 58-year-old patient whose subacute presentation was missed because signals never crossed any single agent's threshold"
        }
    }
    
    with open("circ_l4_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("Results exported to circ_l4_results.json")
    print("=" * 70)
