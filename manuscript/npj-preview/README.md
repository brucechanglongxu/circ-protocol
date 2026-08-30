# npj Digital Medicine preview

This directory contains a lightweight submission-style rendering of the canonical CIRC manuscript.
It approximates the visual structure of an npj Digital Medicine article with a full-width title and
abstract, two-column body, compact typography, and full-width tables and figures. It does not claim
to be the journal's official production template.

Build from `manuscript/` with:

```bash
make npj
```

The output is `npj-preview/circ_npj_digital_medicine_preview.pdf`. The bibliography symlink keeps the
preview tied to the canonical `references.bib`; substantive manuscript revisions must still be
applied to both LaTeX sources until the preview is regenerated.