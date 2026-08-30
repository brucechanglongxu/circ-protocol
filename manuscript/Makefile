.PHONY: pdf clean milestone

VERSION ?= draft

pdf:
	mkdir -p build revisions
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
	cp build/main.pdf main.pdf

milestone: pdf
	cp build/main.pdf revisions/$(VERSION).pdf

clean:
	latexmk -C -outdir=build main.tex
	rm -f main.pdf
