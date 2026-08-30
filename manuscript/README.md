# CIRC manuscript revision workspace

`main.tex` is the active manuscript source. Each reviewer-driven milestone is compiled and copied
to `revisions/` before the next conceptual edit.

```bash
make pdf
make milestone VERSION=00_reconstructed
```

Revision 0 reconstructs the supplied manuscript text. The original simulation images were not
included in the source material, so their captions are retained in marked placeholders. Later
milestones replace those figures rather than attempting to recreate unsupported quantitative plots.
