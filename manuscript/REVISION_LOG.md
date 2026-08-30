# Revision log

## Revision 0: editable reconstruction

- Reconstructed the supplied manuscript as a self-contained LaTeX source.
- Preserved the submitted conceptual structure, four levels, five primitives, vignette, and figure
  captions.
- Corrected only obvious transcription artifacts that would prevent professional typesetting.
- Marked the three unavailable original figures as placeholders.

## Planned revision sequence

1. Reframe CIRC as a non-exhaustive taxonomy plus independent governance levers.
2. Replace the maturity-level structure and remove clinical-prediction claims.
3. Add a balanced standards and alternative-architecture comparison.
4. Add coordination-layer failure modes, degraded operation, and operational human oversight.
5. Replace the oncology vignette and all simulation figures.
6. Add regulatory positioning, structured-message example, definitions, and Ripple Effect scope.
7. Audit claims, references, repetition, paragraph length, and response-to-reviewers coverage.

## Revision 1: risks and independent levers

- Recast CIRC as a conceptual framework rather than a standard, implementation, or certification
  scheme.
- Defined the clinical-agent scope and distinguished administrative, supervised clinical, and
  clinically consequential agents.
- Replaced four cumulative levels with three non-exhaustive risk domains and seven independent
  governance levers.
- Removed the oncology harm narrative, population-level hemorrhage prediction, and all three toy
  simulation figures.
- Added CliniCARE-Bench as an external worked example of trace-observable agent-local risk while
  stating that it does not validate CIRC or test multi-agent coordination.
- Added a staged evaluation agenda and explicit statements that the levers are not necessary,
  sufficient, minimal, or unique to CIRC.

## Revision 2: standards and architecture alternatives

- Replaced the claim that existing infrastructure is broadly insufficient with a capability-level
  account of what FHIR, SMART, CDS Hooks, A2A, MCP, and workflow engines already provide.
- Added a matrix separating current capabilities, extension or profile opportunities, and remaining
  semantics, authorization, workflow, governance, and portability work.
- Corrected the discussion of FHIR Task, Provenance, AuditEvent, Subscription, SMART Backend
  Services, A2A authorization and task lifecycles, and MCP host-controlled isolation.
- Positioned CIRC as an architecture-neutral requirements framework rather than a replacement
  protocol.
- Compared central orchestration, EHR-native workflows, policy engines, supervisory agents,
  distributed protocols, and human-led coordination.

## Revision 3: failure model and operational oversight

- Defined the coordination layer as a potential common-mode failure source rather than only a safety
  mechanism.
- Added identity, state and messaging, dependency, resource-arbitration, coordination-service, and
  human-control failure surfaces with explicit containment behavior.
- Specified action-class safe modes for read-only, reversible administrative, consequential clinical,
  and emergency workflows.
- Clarified federated identity, trust anchors, revocation, and the distinction between attribution
  and clinical authority.
- Replaced generic ``human review'' language with an operational escalation contract covering
  ownership, acknowledgement, deadlines, fallback, suspension authority, workload, and closure.
- Added measurable human-response endpoints and sociotechnical and automation references.
