.PHONY: pdf clean

PDF = circ_npj_digital_medicine_preview.pdf

pdf:
	mkdir -p build
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
	cp build/main.pdf $(PDF)

clean:
	latexmk -C -outdir=build main.tex
	rm -rf build $(PDF)