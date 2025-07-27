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

Currently our protocol dispatcher:

```python
def route_agent(patient_path):
  ...
  if code.startswith("C50"): # breast cancer -> HER2 agent
  elif code.startswith ("E11") # diabetes -> diabetes agent
```

This clinical routing should be offloaded to a config file, and handle fallbacks/multi-agent orchestration and temporal escalation. We should also include permission checks based on identity. Each agent should only act within scope defined by our protocol i.e. knows what clinical tasks it is bounded to, logs only certain recommendations (but does not write to EHR), and we should have permissions restricted via. protocol metadata.

```yaml
agent_id: circ.her2_agent
capabilities:
- read.fhir.Patient
- read.fhir.Condition
- recommend.treatment.her2_pathway
permissions:
  restricted_to: oncologist-reviewed cases
```

**Level 1:** Read-only FHIR API access (similar to MCP)
**Level 2:** Agent-to-agent communication e.g. diabetes agent hands off to Optho agent.
**Level 3:** Universal coordination across agents, platforms.
**Level 4:** Indirect signals and crowd dynamics (alert when 100 patients with same condition request imaging). 

## CIRC Protocol Levels (L1–L4)

| **CIRC-L1** | Direct Tool Invocation           | Agent uses a single data interface (e.g., FHIR API) to retrieve static patient data and act.     | HL7 FHIR + MCP (Model Context Protocol)   | Agent pulls HER2 result and recommends imaging. No coordination with other agents.   |
| **CIRC-L2** | Agent-to-Agent Coordination      | Agents communicate directly to coordinate local actions. Schema and protocol must be aligned.    | A2A (Agent-to-Agent RPC)                  | HER2 agent tells scheduling agent to find imaging before oncology consult.           |
| **CIRC-L3** | Cross-Agent Interoperability     | Agents interoperate across platforms using a shared adapter layer to abstract schema differences.| UAP (Universal Adapter Protocol)          | HER2, diabetes, and referral agents coordinate despite running on different systems. |
| **CIRC-L4** | Crowd-Aware Routing              | Agents respond to indirect signals about system load, trends, or referral surges.                | REP (Ripple Effect Protocol)              | Agent detects Friday overload and rebooks patient for Thursday to reduce congestion. |


# Next Steps
1. We need a `protocol/` directory with `permissions.yaml` / `capabilities.yaml` describing what each agent can do.
2. We need `routing_rules.yaml` and `coordination_rules.json` to have ICD-based orchestration.
3. We need to have an agent registry that maps agents to capabilities.
4. Permissions manager: "Is an agent authorized to recommend therapy", "Should this escalate to higher order agent? Or Physician?" 
