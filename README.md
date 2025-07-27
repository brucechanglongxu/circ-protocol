# CIRC: Clinical Interpretatbility and Routing for Coordination

CIRC: A protocol layer for coordinating clinical agents across systems, specialties, and institutions. A protocol layer for deploying, coordinating, and governing autonomous AI agents in healthcare. CIRC enables clinical agents to route tasks, interoperate across systems (EHRs, claims, labs), and coordinate across specialties. 

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
