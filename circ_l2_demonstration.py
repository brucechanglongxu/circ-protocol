"""
CIRC-L2 Demonstration: Vendor-Bounded Coordination

This simulation demonstrates L2's improvement over L1, and its limitation
that motivates L3.

L2 Key Feature: "Agents can communicate directly with agents they know about,
exchanging structured messages about proposed actions, constraints, and 
dependencies. This requires agents to be designed to interoperate—typically 
meaning they share a vendor or have been explicitly integrated."

Scenario: 
- 3 vendors (A, B, C), each with scheduling + specialty agents
- 180 TCHP patients distributed across vendors based on their care sites
- Some patients have cross-vendor care (oncology at Vendor A, diabetes at Vendor B)

L1: All agents isolated → collisions, no coordination
L2: Same-vendor agents coordinate → reduced collisions within vendor
    But cross-vendor patients still experience L1-level failures

From paper: "120 patients under Vendor A benefit from improved handoffs; 
the remainder experience the same fragmented care as Level 1"
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set
from enum import Enum
from collections import defaultdict
import json

random.seed(42)
np.random.seed(42)

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Patient:
    """Patient with potentially cross-vendor care"""
    id: str
    primary_vendor: str  # Where oncology is managed
    diabetes_vendor: Optional[str]  # Where diabetes is managed (if diabetic)
    has_diabetes: bool
    glucose_optimized: bool
    priority: int
    
    @property
    def is_cross_vendor(self) -> bool:
        """Patient has care spanning multiple vendors"""
        if not self.has_diabetes:
            return False
        return self.diabetes_vendor != self.primary_vendor

@dataclass
class ImagingSlot:
    id: str
    time: str
    status: str = "available"
    booked_for: Optional[str] = None
    booked_by: Optional[str] = None

@dataclass
class BookingRequest:
    agent_id: str
    vendor: str
    patient_id: str
    preferred_slot: str
    priority: int
    timestamp: float
    requires_diabetes_clearance: bool = False

@dataclass
class CoordinationMessage:
    """L2 structured message between same-vendor agents"""
    sender: str
    receiver: str
    message_type: str  # "propose", "acknowledge", "block", "notify"
    patient_id: str
    action: str
    constraints: dict = field(default_factory=dict)

# =============================================================================
# L1 Environment: No Coordination
# =============================================================================

class L1Environment:
    """CIRC-L1: Agents operate in complete isolation"""
    
    def __init__(self, num_slots: int = 30):
        self.slots = {}
        for i in range(num_slots):
            hour = 8 + (i // 4)
            minute = (i % 4) * 15
            slot_id = f"FRI-{hour:02d}{minute:02d}"
            self.slots[slot_id] = ImagingSlot(id=slot_id, time=f"{hour:02d}:{minute:02d}")
        
        self.collision_log = []
        self.booking_log = []
        self.prerequisite_violations = []  # Diabetic patients booked without clearance
        
    def get_available_slots(self) -> List[ImagingSlot]:
        return [s for s in self.slots.values() if s.status == "available"]
    
    def submit_booking(self, request: BookingRequest) -> tuple:
        slot = self.slots.get(request.preferred_slot)
        
        if slot is None:
            return False, "INVALID_SLOT"
        
        if slot.status == "booked":
            self.collision_log.append({
                "agent": request.agent_id,
                "patient": request.patient_id,
                "slot": request.preferred_slot,
                "vendor": request.vendor
            })
            return False, "COLLISION"
        
        # L1: No prerequisite checking - diabetes agent not consulted
        if request.requires_diabetes_clearance:
            self.prerequisite_violations.append({
                "patient": request.patient_id,
                "reason": "Diabetes clearance not obtained"
            })
        
        slot.status = "booked"
        slot.booked_for = request.patient_id
        slot.booked_by = request.agent_id
        
        self.booking_log.append({
            "patient": request.patient_id,
            "slot": request.preferred_slot,
            "vendor": request.vendor
        })
        
        return True, "BOOKED"

# =============================================================================
# L2 Environment: Vendor-Bounded Coordination
# =============================================================================

class L2Environment:
    """
    CIRC-L2: Direct agent-to-agent communication within vendor boundaries.
    
    Key features:
    - Same-vendor agents can exchange structured messages
    - Agents notify each other before booking
    - Prerequisite checking WITHIN vendor
    - BUT cross-vendor coordination doesn't exist
    """
    
    def __init__(self, num_slots: int = 30):
        self.slots = {}
        for i in range(num_slots):
            hour = 8 + (i // 4)
            minute = (i % 4) * 15
            slot_id = f"FRI-{hour:02d}{minute:02d}"
            self.slots[slot_id] = ImagingSlot(id=slot_id, time=f"{hour:02d}:{minute:02d}")
        
        # Vendor-specific coordination state
        self.vendor_pending = defaultdict(dict)  # vendor -> {slot_id -> request}
        self.vendor_diabetes_agents = {}  # vendor -> diabetes_agent
        
        self.collision_log = []
        self.booking_log = []
        self.prerequisite_violations = []
        self.coordination_messages = []
        self.within_vendor_coordinations = 0
        self.cross_vendor_gaps = []
        
    def register_diabetes_agent(self, vendor: str, agent_id: str):
        """Register diabetes agent for a vendor"""
        self.vendor_diabetes_agents[vendor] = agent_id
        
    def get_available_slots(self) -> List[ImagingSlot]:
        return [s for s in self.slots.values() if s.status == "available"]
    
    def get_vendor_available_slots(self, vendor: str) -> List[ImagingSlot]:
        """L2: Vendor agents share availability state"""
        pending_slots = set(self.vendor_pending.get(vendor, {}).keys())
        return [s for s in self.slots.values() 
                if s.status == "available" and s.id not in pending_slots]
    
    def propose_booking(self, request: BookingRequest, patient: Patient) -> tuple:
        """
        L2: Structured booking proposal with same-vendor coordination.
        """
        slot = self.slots.get(request.preferred_slot)
        
        if slot is None:
            return False, "INVALID_SLOT", None
        
        if slot.status == "booked":
            self.collision_log.append({
                "agent": request.agent_id,
                "patient": request.patient_id,
                "slot": request.preferred_slot,
                "vendor": request.vendor,
                "type": "slot_taken"
            })
            return False, "COLLISION", None
        
        # Check if another same-vendor agent has pending request
        if request.vendor in self.vendor_pending:
            if request.preferred_slot in self.vendor_pending[request.vendor]:
                # L2: Same-vendor conflict detected BEFORE booking
                # Negotiate based on priority
                other_request = self.vendor_pending[request.vendor][request.preferred_slot]
                
                self.coordination_messages.append(CoordinationMessage(
                    sender=request.agent_id,
                    receiver=other_request.agent_id,
                    message_type="negotiate",
                    patient_id=request.patient_id,
                    action="slot_conflict",
                    constraints={"slot": request.preferred_slot}
                ))
                
                self.within_vendor_coordinations += 1
                
                # Higher priority wins
                if request.priority < other_request.priority:
                    # Current request wins, notify other agent
                    del self.vendor_pending[request.vendor][request.preferred_slot]
                else:
                    # Other request wins
                    return False, "NEGOTIATION_LOST", other_request.agent_id
        
        # L2: Check prerequisites with same-vendor diabetes agent
        if request.requires_diabetes_clearance:
            if patient.diabetes_vendor == request.vendor:
                # Same vendor - can coordinate
                diabetes_agent = self.vendor_diabetes_agents.get(request.vendor)
                if diabetes_agent:
                    self.coordination_messages.append(CoordinationMessage(
                        sender=request.agent_id,
                        receiver=diabetes_agent,
                        message_type="check_prerequisites",
                        patient_id=request.patient_id,
                        action="imaging_clearance",
                        constraints={"glucose_required": True}
                    ))
                    self.within_vendor_coordinations += 1
                    
                    if not patient.glucose_optimized:
                        # Same-vendor diabetes agent blocks
                        return False, "BLOCKED_PREREQUISITES", diabetes_agent
            else:
                # Cross-vendor - NO coordination possible at L2
                # This is the key L2 limitation!
                self.cross_vendor_gaps.append({
                    "patient": request.patient_id,
                    "scheduling_vendor": request.vendor,
                    "diabetes_vendor": patient.diabetes_vendor,
                    "issue": "Cannot coordinate across vendor boundary"
                })
                
                if not patient.glucose_optimized:
                    # Prerequisite violation - diabetes agent not consulted
                    self.prerequisite_violations.append({
                        "patient": request.patient_id,
                        "reason": "Cross-vendor: diabetes agent not consulted",
                        "scheduling_vendor": request.vendor,
                        "diabetes_vendor": patient.diabetes_vendor
                    })
        
        # Mark as pending for this vendor
        if request.vendor not in self.vendor_pending:
            self.vendor_pending[request.vendor] = {}
        self.vendor_pending[request.vendor][request.preferred_slot] = request
        
        return True, "PENDING", None
    
    def confirm_booking(self, request: BookingRequest) -> tuple:
        """Confirm a pending booking"""
        slot = self.slots.get(request.preferred_slot)
        
        if slot.status == "booked":
            return False, "COLLISION"
        
        slot.status = "booked"
        slot.booked_for = request.patient_id
        slot.booked_by = request.agent_id
        
        # Clear pending
        if request.vendor in self.vendor_pending:
            if request.preferred_slot in self.vendor_pending[request.vendor]:
                del self.vendor_pending[request.vendor][request.preferred_slot]
        
        self.booking_log.append({
            "patient": request.patient_id,
            "slot": request.preferred_slot,
            "vendor": request.vendor
        })
        
        return True, "BOOKED"

# =============================================================================
# Agents
# =============================================================================

class SchedulingAgent:
    """Scheduling agent belonging to a specific vendor"""
    
    def __init__(self, agent_id: str, vendor: str):
        self.id = agent_id
        self.vendor = vendor
        self.patients = []
        
    def add_patient(self, patient: Patient):
        self.patients.append(patient)
    
    def select_optimal_slot(self, available_slots: List[ImagingSlot], patient: Patient) -> Optional[str]:
        """Select slot - all agents prefer afternoon"""
        if not available_slots:
            return None
        
        # Prefer afternoon slots
        afternoon = [s for s in available_slots if int(s.id.split('-')[1][:2]) >= 13]
        target = afternoon if afternoon else available_slots
        
        target.sort(key=lambda s: s.id)
        idx = min(patient.priority - 1, len(target) - 1)
        return target[idx].id


class DiabetesAgent:
    """Diabetes agent belonging to a specific vendor"""
    
    def __init__(self, agent_id: str, vendor: str):
        self.id = agent_id
        self.vendor = vendor
        self.patients = []
    
    def add_patient(self, patient: Patient):
        if patient.has_diabetes and patient.diabetes_vendor == self.vendor:
            self.patients.append(patient)
    
    def check_clearance(self, patient_id: str) -> bool:
        """Check if patient is cleared for imaging"""
        patient = next((p for p in self.patients if p.id == patient_id), None)
        if patient:
            return patient.glucose_optimized
        return True  # Not our patient

# =============================================================================
# Simulation
# =============================================================================

def create_patient_population(n_patients: int = 60) -> List[Patient]:
    """
    Create patient population with cross-vendor care patterns.
    
    Distribution:
    - ~40% single-vendor care (all care at one vendor)
    - ~60% cross-vendor care (oncology and diabetes at different vendors)
    
    This reflects real-world fragmentation described in the paper.
    """
    patients = []
    vendors = ["VendorA", "VendorB", "VendorC"]
    
    for i in range(n_patients):
        primary_vendor = random.choice(vendors)
        has_diabetes = random.random() < 0.5
        
        if has_diabetes:
            # 60% chance of cross-vendor diabetes care
            if random.random() < 0.6:
                other_vendors = [v for v in vendors if v != primary_vendor]
                diabetes_vendor = random.choice(other_vendors)
            else:
                diabetes_vendor = primary_vendor
        else:
            diabetes_vendor = None
        
        patients.append(Patient(
            id=f"PT-{i:03d}",
            primary_vendor=primary_vendor,
            diabetes_vendor=diabetes_vendor,
            has_diabetes=has_diabetes,
            glucose_optimized=random.random() > 0.25 if has_diabetes else True,
            priority=random.choices([1, 2, 3, 4, 5], weights=[5, 15, 30, 30, 20])[0]
        ))
    
    return patients


def run_l1_simulation(patients: List[Patient], n_slots: int = 30) -> Dict:
    """Run L1 simulation - no coordination"""
    env = L1Environment(num_slots=n_slots)
    
    # Create agents for each vendor
    vendors = ["VendorA", "VendorB", "VendorC"]
    sched_agents = {v: SchedulingAgent(f"SCHED-{v}", v) for v in vendors}
    
    # Assign patients to agents
    for patient in patients:
        sched_agents[patient.primary_vendor].add_patient(patient)
    
    # All agents select slots concurrently (L1 failure mode)
    all_requests = []
    for vendor, agent in sched_agents.items():
        available = env.get_available_slots()
        for patient in agent.patients:
            slot_id = agent.select_optimal_slot(available, patient)
            if slot_id:
                all_requests.append(BookingRequest(
                    agent_id=agent.id,
                    vendor=vendor,
                    patient_id=patient.id,
                    preferred_slot=slot_id,
                    priority=patient.priority,
                    timestamp=random.random(),
                    requires_diabetes_clearance=patient.has_diabetes and not patient.glucose_optimized
                ))
    
    # Process in timestamp order
    all_requests.sort(key=lambda r: r.timestamp)
    
    results = {"booked": 0, "collisions": 0, "retries": 0}
    for request in all_requests:
        success, msg = env.submit_booking(request)
        if success:
            results["booked"] += 1
        else:
            results["collisions"] += 1
            # Retry
            for _ in range(2):
                available = env.get_available_slots()
                if not available:
                    break
                results["retries"] += 1
                new_slot = random.choice(available).id
                retry = BookingRequest(
                    agent_id=request.agent_id,
                    vendor=request.vendor,
                    patient_id=request.patient_id,
                    preferred_slot=new_slot,
                    priority=request.priority,
                    timestamp=random.random(),
                    requires_diabetes_clearance=request.requires_diabetes_clearance
                )
                if env.submit_booking(retry)[0]:
                    results["booked"] += 1
                    break
    
    return {
        "booked": results["booked"],
        "collisions": len(env.collision_log),
        "retries": results["retries"],
        "prerequisite_violations": len(env.prerequisite_violations),
        "cross_vendor_gaps": 0  # L1 doesn't track this
    }


def run_l2_simulation(patients: List[Patient], n_slots: int = 30) -> Dict:
    """Run L2 simulation - vendor-bounded coordination"""
    env = L2Environment(num_slots=n_slots)
    
    vendors = ["VendorA", "VendorB", "VendorC"]
    sched_agents = {v: SchedulingAgent(f"SCHED-{v}", v) for v in vendors}
    diabetes_agents = {v: DiabetesAgent(f"DIAB-{v}", v) for v in vendors}
    
    # Register diabetes agents
    for vendor, agent in diabetes_agents.items():
        env.register_diabetes_agent(vendor, agent.id)
    
    # Assign patients
    for patient in patients:
        sched_agents[patient.primary_vendor].add_patient(patient)
        if patient.has_diabetes and patient.diabetes_vendor:
            diabetes_agents[patient.diabetes_vendor].add_patient(patient)
    
    results = {"booked": 0, "blocked": 0, "negotiation_resolved": 0}
    
    # Process by vendor (L2: within-vendor coordination)
    for vendor, agent in sched_agents.items():
        for patient in agent.patients:
            # L2: Use vendor-aware slot selection
            available = env.get_vendor_available_slots(vendor)
            slot_id = agent.select_optimal_slot(available, patient)
            
            if not slot_id:
                continue
            
            request = BookingRequest(
                agent_id=agent.id,
                vendor=vendor,
                patient_id=patient.id,
                preferred_slot=slot_id,
                priority=patient.priority,
                timestamp=random.random(),
                requires_diabetes_clearance=patient.has_diabetes and not patient.glucose_optimized
            )
            
            # L2: Propose with coordination
            success, msg, other = env.propose_booking(request, patient)
            
            if success:
                # Confirm booking
                if env.confirm_booking(request)[0]:
                    results["booked"] += 1
            elif msg == "BLOCKED_PREREQUISITES":
                results["blocked"] += 1
            elif msg == "NEGOTIATION_LOST":
                results["negotiation_resolved"] += 1
                # Try another slot
                available = env.get_vendor_available_slots(vendor)
                if available:
                    new_slot = random.choice(available).id
                    request.preferred_slot = new_slot
                    if env.propose_booking(request, patient)[0]:
                        if env.confirm_booking(request)[0]:
                            results["booked"] += 1
            else:
                # Collision with other vendor - retry
                available = env.get_available_slots()
                if available:
                    new_slot = random.choice(available).id
                    request.preferred_slot = new_slot
                    if env.propose_booking(request, patient)[0]:
                        if env.confirm_booking(request)[0]:
                            results["booked"] += 1
    
    return {
        "booked": results["booked"],
        "collisions": len(env.collision_log),
        "prerequisite_violations": len(env.prerequisite_violations),
        "cross_vendor_gaps": len(env.cross_vendor_gaps),
        "within_vendor_coordinations": env.within_vendor_coordinations,
        "blocked_by_coordination": results["blocked"]
    }


def run_comparative_simulation(n_runs: int = 30, n_patients: int = 60, n_slots: int = 25) -> Dict:
    """Run multiple simulations comparing L1 vs L2"""
    
    l1_metrics = defaultdict(list)
    l2_metrics = defaultdict(list)
    
    # Track by patient type
    l1_cross_vendor = {"collisions": [], "violations": []}
    l2_cross_vendor = {"collisions": [], "violations": []}
    
    for run in range(n_runs):
        patients = create_patient_population(n_patients)
        
        # Count cross-vendor patients
        cross_vendor_patients = [p for p in patients if p.is_cross_vendor]
        single_vendor_patients = [p for p in patients if not p.is_cross_vendor]
        
        l1 = run_l1_simulation(patients.copy(), n_slots)
        l1_metrics["collisions"].append(l1["collisions"])
        l1_metrics["violations"].append(l1["prerequisite_violations"])
        l1_metrics["booked"].append(l1["booked"])
        l1_metrics["retries"].append(l1["retries"])
        
        l2 = run_l2_simulation(patients.copy(), n_slots)
        l2_metrics["collisions"].append(l2["collisions"])
        l2_metrics["violations"].append(l2["prerequisite_violations"])
        l2_metrics["booked"].append(l2["booked"])
        l2_metrics["cross_vendor_gaps"].append(l2["cross_vendor_gaps"])
        l2_metrics["within_vendor_coord"].append(l2["within_vendor_coordinations"])
        l2_metrics["blocked"].append(l2["blocked_by_coordination"])
    
    return {
        "n_runs": n_runs,
        "n_patients": n_patients,
        "n_slots": n_slots,
        "L1": {k: (np.mean(v), np.std(v)) for k, v in l1_metrics.items()},
        "L2": {k: (np.mean(v), np.std(v)) for k, v in l2_metrics.items()}
    }


def create_figure(results: Dict):
    """Create publication figure for L2 demonstration"""
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    
    L1_COLOR = '#E74C3C'  # Red
    L2_COLOR = '#3498DB'  # Blue
    PROBLEM_COLOR = '#F39C12'  # Orange for remaining problems
    
    # Panel A: Collision Reduction
    ax = axes[0]
    
    l1_collisions = results["L1"]["collisions"][0]
    l2_collisions = results["L2"]["collisions"][0]
    
    bars = ax.bar([0, 1], [l1_collisions, l2_collisions], 
                  color=[L1_COLOR, L2_COLOR], alpha=0.8, width=0.6)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['L1\n(Isolated)', 'L2\n(Vendor-Bounded)'])
    ax.set_ylabel('Slot Collisions per Cycle')
    ax.set_title('A. Resource Collisions', fontweight='bold')
    
    # Improvement annotation
    if l1_collisions > 0:
        reduction = (l1_collisions - l2_collisions) / l1_collisions * 100
        ax.annotate(f'{reduction:.0f}% reduction\n(within-vendor\ncoordination)', 
                   xy=(0.5, (l1_collisions + l2_collisions)/2),
                   ha='center', fontsize=10, color='#27AE60', fontweight='bold')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
               f'{height:.1f}', ha='center', va='bottom', fontsize=11)
    
    # Panel B: Prerequisite Violations (the L2 limitation)
    ax = axes[1]
    
    l1_violations = results["L1"]["violations"][0]
    l2_violations = results["L2"]["violations"][0]
    l2_blocked = results["L2"]["blocked"][0]
    
    x = np.arange(2)
    width = 0.35
    
    # L1: All violations (unsafe bookings)
    bars1 = ax.bar(0, l1_violations, width=0.5, label='Unsafe bookings\n(no coordination)', 
                   color=L1_COLOR, alpha=0.8)
    
    # L2: Split into caught (within-vendor) and missed (cross-vendor)
    bars2 = ax.bar(1 - width/2, l2_blocked, width, label='Caught\n(same-vendor)', 
                   color='#27AE60', alpha=0.8)
    bars3 = ax.bar(1 + width/2, l2_violations, width, label='Missed\n(cross-vendor)', 
                   color=PROBLEM_COLOR, alpha=0.8)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['L1\n(Isolated)', 'L2\n(Vendor-Bounded)'])
    ax.set_ylabel('Prerequisite Violations per Cycle')
    ax.set_title('B. Diabetic Patient Safety', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    # Annotation for L2 limitation
    if l2_violations > 0:
        ax.annotate(f'{l2_violations:.1f} cross-vendor\ngaps remain', 
                   xy=(1 + width/2, l2_violations + 0.3),
                   ha='center', fontsize=9, color='#D35400', fontweight='bold')
    
    # Panel C: The Asymmetry Problem
    ax = axes[2]
    
    # Calculate coordination events
    l2_coord = results["L2"]["within_vendor_coord"][0]
    l2_gaps = results["L2"]["cross_vendor_gaps"][0]
    
    categories = ['Within-Vendor\nCoordination', 'Cross-Vendor\nGaps']
    values = [l2_coord, l2_gaps]
    colors = ['#27AE60', PROBLEM_COLOR]
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.6)
    
    ax.set_ylabel('Events per Cycle')
    ax.set_title('C. L2 Coordination Asymmetry', fontweight='bold')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
               f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Key insight annotation
    ax.text(0.5, max(values) * 0.4, 
           '"Patients do not organize\ntheir diseases by vendor"',
           ha='center', fontsize=10, style='italic', color='#7F8C8D',
           transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('circ_l2_demo.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('circ_l2_demo.pdf', bbox_inches='tight')
    
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CIRC-L2 Demonstration: Vendor-Bounded Coordination")
    print("=" * 70)
    print()
    
    print("Running simulation...")
    print("  - 60 patients across 3 vendors")
    print("  - ~50% diabetic, ~60% of diabetics have cross-vendor care")
    print("  - 25 imaging slots (creates contention)")
    print()
    
    results = run_comparative_simulation(n_runs=30, n_patients=60, n_slots=25)
    
    print("RESULTS")
    print("-" * 70)
    print(f"{'Metric':<40} {'L1 (Isolated)':<15} {'L2 (Vendor)':<15}")
    print("-" * 70)
    print(f"{'Slot Collisions':<40} {results['L1']['collisions'][0]:<15.1f} {results['L2']['collisions'][0]:<15.1f}")
    print(f"{'Prerequisite Violations (unsafe)':<40} {results['L1']['violations'][0]:<15.1f} {results['L2']['violations'][0]:<15.1f}")
    print(f"{'Successfully Booked':<40} {results['L1']['booked'][0]:<15.1f} {results['L2']['booked'][0]:<15.1f}")
    print(f"{'Blocked by Same-Vendor Coordination':<40} {'-':<15} {results['L2']['blocked'][0]:<15.1f}")
    print(f"{'Cross-Vendor Gaps (L2 limitation)':<40} {'-':<15} {results['L2']['cross_vendor_gaps'][0]:<15.1f}")
    print("-" * 70)
    print()
    
    # Calculate improvements and remaining gaps
    collision_reduction = (results['L1']['collisions'][0] - results['L2']['collisions'][0]) / results['L1']['collisions'][0] * 100
    violation_reduction = (results['L1']['violations'][0] - results['L2']['violations'][0]) / results['L1']['violations'][0] * 100 if results['L1']['violations'][0] > 0 else 0
    
    print("L2 IMPROVEMENTS OVER L1:")
    print(f"  ✓ Collision reduction: {collision_reduction:.0f}%")
    print(f"  ✓ Same-vendor prerequisite checks: {results['L2']['blocked'][0]:.1f} unsafe bookings prevented")
    print()
    
    print("L2 REMAINING LIMITATIONS (motivates L3):")
    print(f"  ✗ Cross-vendor gaps: {results['L2']['cross_vendor_gaps'][0]:.1f} coordination failures")
    print(f"  ✗ Cross-vendor violations: {results['L2']['violations'][0]:.1f} unsafe bookings still occur")
    print(f"  ✗ Asymmetric outcomes: patients with cross-vendor care disadvantaged")
    print()
    
    print("Generating figure...")
    create_figure(results)
    print("Figure saved as circ_l2_demo.png and .pdf")
    print()
    
    # Export JSON
    output = {
        "scenario": "Vendor-Bounded Coordination",
        "paper": "Asimov's Laws for Clinical AI Agents (Xu, Chopra, Ryu 2025)",
        "l2_capability": "Direct agent-to-agent communication within vendor boundaries",
        "parameters": {
            "n_runs": results["n_runs"],
            "n_patients": results["n_patients"],
            "n_slots": results["n_slots"],
            "diabetes_rate": 0.5,
            "cross_vendor_rate": 0.6
        },
        "results": {
            "L1": {
                "collisions_mean": results["L1"]["collisions"][0],
                "violations_mean": results["L1"]["violations"][0],
                "booked_mean": results["L1"]["booked"][0]
            },
            "L2": {
                "collisions_mean": results["L2"]["collisions"][0],
                "violations_mean": results["L2"]["violations"][0],
                "booked_mean": results["L2"]["booked"][0],
                "blocked_mean": results["L2"]["blocked"][0],
                "cross_vendor_gaps_mean": results["L2"]["cross_vendor_gaps"][0],
                "within_vendor_coord_mean": results["L2"]["within_vendor_coord"][0]
            }
        },
        "improvements": {
            "collision_reduction_pct": collision_reduction,
            "same_vendor_violations_caught": results["L2"]["blocked"][0]
        },
        "limitations": {
            "cross_vendor_gaps": results["L2"]["cross_vendor_gaps"][0],
            "cross_vendor_violations": results["L2"]["violations"][0],
            "clinical_interpretation": "Patients with care fragmented across vendors experience L1-level coordination failures"
        }
    }
    
    with open("circ_l2_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("Results exported to circ_l2_results.json")
    print("=" * 70)
