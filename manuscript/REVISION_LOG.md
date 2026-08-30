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

## Revision 4: structured message and architecture

- Added operational definitions for agent, intent, authority, coordination state, and evidence.
- Added a non-normative `circ.intent.v0` scheduling message with identity, scope, clinical context,
  preconditions, reversibility, escalation, and provenance fields.
- Explained how the message can map to FHIR, A2A, or local workflow representations without
  defining a competing wire protocol.
- Added an architecture figure showing institutional boundaries, coordination, identity, policy,
  audit, clinical-state, and human-oversight paths.
- Kept the design architecture-neutral by treating the illustrated coordination service as one
  replaceable implementation pattern.

## Revision 5: regulatory boundaries and Ripple Effect scope

- Clarified that the Ripple Effect Protocol motivates a population-level failure pattern but does
  not validate CIRC, establish clinical transfer, or define a required CIRC mechanism.
- Positioned regulatory status at the level of each software function, intended use, actor role, and
  deployment context rather than the agent label or system topology.
- Updated the U.S. discussion with final FDA guidance on clinical decision support and predetermined
  change control plans for AI-enabled device software functions.
- Distinguished CIRC evidence from classification, premarket or conformity assessment,
  institutional deployment decisions, and post-market change control.
- Added a deployment-dossier and impact-assessment rule for changes to models, actions,
  dependencies, trust boundaries, populations, resource policies, and human fallbacks.

## Revision 6: reviewer closure

- Added a lever-by-lever crosswalk linking existing infrastructure, residual coordination contracts,
  gap types, implementation alternatives, and observable evidence.
- Explained the functional derivation of the seven levers while retaining explicit limits on their
  necessity, sufficiency, and completeness.
- Mapped U.S. and European regulatory duties to CIRC operational records and separated those records
  from classification, authorization, conformity, liability, and local-law decisions.
- Removed obsolete vignette and superseded regulatory references, added workflow and systems-safety
  literature, and cited every retained bibliography entry in the manuscript.
- Completed a manuscript-wide claims and prose audit and synchronized the canonical and npj sources.
- Added an accurate point-by-point response covering all three reviewers and the limits of the
  manuscript's evidentiary claims.

## Revision 7: npj Digital Medicine resubmission package

- Replaced the title with a seven-word literal title without punctuation and reduced the abstract to
  69 words.
- Reduced the main prose to 2,961 words while preserving the reviewer-driven standards, architecture,
  failure, oversight, regulatory, and evidence content.
- Added the corresponding-author email, acknowledgments, and an author-approved contribution
  statement for all three authors.
- Added editable and PDF cover-letter and point-by-point response documents.
- Added a reproducible submission-package target containing the presentable manuscript PDF, editable
  source and dependencies, editorial documents, and upload checklist.
