# CIRC: Clinical Infrastructure for Reliable Coordination

CIRC (Clinical Infrastructure for Reliable Coordination) is a conceptual governance framework for coordinating clinical AI agents across systems, specialties, and institutions. It focuses on how agents initiate and route tasks while respecting clinical scope, data permissions, and institutional governance. This repository contains early protocol demonstrations and trace-based experiments for evaluating those controls.

## Trace governance experiments

The current empirical work treats scope enforcement, dependency completeness, and citation
provenance as independent governance levers rather than sequential maturity levels. The
[trace-governance study](experiments/trace-governance/README.md) provides a preregistered matched
replay design built on the public CliniCARE-Bench scenarios, a complete risk codebook for its 104
prohibited process shortcuts, and a protected-data runner for MIMIC-authorized environments.

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

## Legacy coordination-level demonstrations

The following levels describe the original demonstration scripts and are not used as a cumulative
maturity model in the trace-governance study.

**Level 1:** Read-only FHIR API access (similar to MCP)
**Level 2:** Agent-to-agent communication e.g. diabetes agent hands off to Optho agent.
**Level 3:** Universal coordination across agents, platforms.
**Level 4:** Indirect signals and crowd dynamics (alert when 100 patients with same condition request imaging). 


# Next Steps
1. We need a `protocol/` directory with `permissions.yaml` / `capabilities.yaml` describing what each agent can do.
2. We need `routing_rules.yaml` and `coordination_rules.json` to have ICD-based orchestration.
3. We need to have an agent registry that maps agents to capabilities.
4. Permissions manager: "Is an agent authorized to recommend therapy", "Should this escalate to higher order agent? Or Physician?" 
