# Response to reviewers

**Manuscript:** Governance Requirements for Coordinating Clinical AI Agents

Thank you for the detailed reviews. We revised the manuscript throughout. We replaced the four-level
model with a risk taxonomy and independent governance controls, removed the simulations and clinical
vignette, and added comparisons with existing standards, alternative architectures, failure modes,
human oversight, and regulatory boundaries. The revised paper does not claim empirical validation of
CIRC. The headings below paraphrase the reviewer comments.

## Reviewer 1

### Comment 1: Clarify the central contribution and the relationship between individual-agent and multi-agent safety

**Response:** The Introduction now defines interaction risk and separates agent-local error from
failures involving shared state, dependencies, handoffs, resources, and institutional response. Table
2 uses the same three-part distinction.

### Comment 2: The four CIRC levels imply an unvalidated progression or maturity hierarchy

**Response:** We removed the four levels. Table 3 presents seven independent controls that deployments
can select according to risk. The text states that they are neither necessary nor sufficient and do
not form a maturity sequence.

### Comment 3: The simulations require methodological support and should not be presented as evidence of effectiveness

**Response:** We removed all three simulations, their figures, and their numerical claims. Section 8
now separates risk observability, control performance, and clinical or operational outcomes. The paper
reports no evidence that CIRC improves outcomes.

### Comment 4: Ground the framework in concrete, measurable behavior

**Response:** Section 4.1 defines interaction risk, collision, scope violation, dependency violation,
and failure containment. It also includes a non-normative coordination message. Table 3 links each
control to observable evidence.

## Reviewer 2

### Comment 1: CIRC is presented inconsistently as a taxonomy, maturity model, architecture, protocol, and governance framework

**Response:** We now describe CIRC consistently as a conceptual, architecture-neutral governance
framework. It links a non-exhaustive risk taxonomy to controls and evidence requirements; it is not a
wire protocol, implemented architecture, certification scheme, or standard.

### Comment 2: The manuscript does not establish why the proposed controls are distinct from existing standards and infrastructure

**Response:** Tables 1 and 3 compare CIRC with FHIR, SMART on FHIR, CDS Hooks, A2A, MCP, and workflow
and policy engines. They distinguish existing capabilities from residual semantics, authorization,
workflow, governance, and portability questions. The text states that existing infrastructure may
implement the controls without a new transport protocol.

### Comment 3: The seven levers require a derivation and should not be claimed as minimal or sufficient

**Response:** Section 4 explains that the seven controls come from operational questions about
identity, permission, state, dependencies, shared resources, responsibility, and containment. We
describe this as a functional starting point, not an empirical derivation or closed ontology.
Privacy, transport security, clinical validation, and model performance remain separate obligations.

### Comment 4: Central orchestration and other architectures may address the same risks without direct agent-to-agent communication

**Response:** Table 4 compares central orchestration, EHR-native workflows, policy and identity
layers, supervisory agents, distributed protocols, and human-led coordination. Figure 2 is labeled as
one possible implementation. The manuscript does not prefer decentralized communication.

### Comment 5: The coordination layer creates its own common-mode and security risks

**Response:** Section 5 and Table 5 cover compromised agents, stale or replayed messages, incorrect
dependencies, deadlock, retry amplification, misleading resource signals, service outage, network
partition, and failure of the human control path. The section also defines action-specific degraded
modes, revocation, recovery, and return of control to a named role.

### Comment 6: Human review is not an operational control unless ownership and non-response behavior are specified

**Response:** Section 6 replaces generic human review with an escalation contract specifying
ownership, trigger, requested action, deadline, action state, fallback, and closure. It also covers
suspension authority, separation of duties, alert burden, and response measures.

### Comment 7: The oncology vignette and population-level prediction claim extend beyond coordination governance

**Response:** We removed the oncology vignette and population-level clinical prediction claim.
Resource awareness now concerns capacity, demand, priority, concentration, fairness, and common-mode
failure. The limitations state that clinical prediction would require a separate validated model.

## Reviewer 3

### Comment 1: Define the actors, terms, and proposed information exchange more precisely

**Response:** The Introduction defines a clinical agent and distinguishes administrative, supervised
clinical, and clinically consequential agents. Section 4.1 defines the main failure terms and gives a
non-normative message that can map to FHIR, A2A, or a local workflow schema.

### Comment 2: Explain the relationship between CIRC and the Ripple Effect Protocol

**Response:** Section 2 now states that the Ripple Effect Protocol motivates a possible failure
pattern but does not validate CIRC or establish clinical transfer. CIRC does not require sensitivity
sharing, and any aggregate signal would need separate evaluation.

### Comment 3: Map CIRC to concrete regulatory duties and identify what the framework does not resolve

**Response:** Section 7 addresses regulation by software function, intended use, statutory role, and
deployment context. It covers U.S. clinical decision support and change-control guidance and relevant
EU AI Act duties. Table 6 separates existing obligations, records CIRC could support, and questions
CIRC cannot resolve.

### Comment 4: Strengthen the scientific and standards grounding and remove unsupported claims

**Response:** We removed references used only for deleted material or superseded guidance and cited
every retained entry. We added sources on workflow, state-machine replication, zero trust,
sociotechnical safety, automation, and systems safety. We also narrowed claims about novelty,
necessity, sufficiency, validation, clinical benefit, and regulatory conformity.
