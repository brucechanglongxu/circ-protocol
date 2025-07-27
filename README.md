# CIRC: Clinical Interpretatbility and Routing for Coordination

CIRC (Clinical Interoperability and Routing for Coordination) is a lightweight protocol for integrating AI agents into real-world clinical systems, for coordinating clinical agents across systems, specialties, and institutions. CIRC enables clinical agents to route tasks, interoperate across systems (EHRs, claims, labs) and coordinate across specialties. It focuses on how agents initiate and route tasks—like drafting treatment plans, submitting prior authorizations, or retrieving diagnostic data—while respecting clinical scope, data permissions, and institutional governance. Rather than treating agents as black-box tools, CIRC frames them as structured participants in care delivery. Each agent operates within a defined task boundary and flows through a routing layer that enforces oversight, logs actions, and supports real-time auditability. This makes CIRC agents easier to trust, compose, and regulate inside health systems.

This matters because most healthcare infrastructure wasn’t built with autonomous agents in mind. Standards like FHIR and HL7 support data exchange, but not continuous, real-time coordination between AI services. To address this, CIRC builds on existing protocols but introduces practical layers to manage task-level autonomy, increasing the agent’s _“circle of influence”_ only when supervision and safeguards allow.

# Example Use Case

**Agent: HER2 Pathway Agent**
Input
```json
{
  "patient_id": "ABC123",
  "her2_status": "positive",
  "stage": "early",
  "hormone_status": "ER+ / PR+"
}
```
Output
```CSS
[✔] CIRC authorized this request.
[→] Routing to her2_pathway_agent...
[✓] Recommendation: Trastuzumab + Pertuzumab + Endocrine Therapy.
```
