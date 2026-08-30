.PHONY: pdf clean milestone npj editorial submission

VERSION ?= draft
PACKAGE_DIR = submission-package
PACKAGE_ZIP = CIRC_npjDM_resubmission_package.zip
FINAL_MANUSCRIPT_PDF = CIRC_npjDM_manuscript.pdf

pdf:
	mkdir -p build revisions
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
	cp build/main.pdf main.pdf

milestone: pdf
	cp build/main.pdf revisions/$(VERSION).pdf

npj:
	$(MAKE) -C npj-preview pdf

editorial:
	mkdir -p build/cover-letter build/response
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/cover-letter COVER_LETTER.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/response RESPONSE_TO_REVIEWERS.tex
	cp build/cover-letter/COVER_LETTER.pdf COVER_LETTER.pdf
	cp build/response/RESPONSE_TO_REVIEWERS.pdf RESPONSE_TO_REVIEWERS.pdf

submission: pdf npj editorial
	rm -rf $(PACKAGE_DIR)
	mkdir -p $(PACKAGE_DIR)/manuscript $(PACKAGE_DIR)/figures $(PACKAGE_DIR)/editorial
	cp npj-preview/circ_npj_digital_medicine_preview.pdf $(FINAL_MANUSCRIPT_PDF)
	cp $(FINAL_MANUSCRIPT_PDF) $(PACKAGE_DIR)/CIRC_npjDM_manuscript.pdf
	cp npj-preview/main.tex $(PACKAGE_DIR)/manuscript/main.tex
	cp references.bib $(PACKAGE_DIR)/manuscript/references.bib
	cp figures/circ_architecture.tex $(PACKAGE_DIR)/figures/circ_architecture.tex
	cp COVER_LETTER.pdf $(PACKAGE_DIR)/CIRC_npjDM_cover_letter.pdf
	cp RESPONSE_TO_REVIEWERS.pdf $(PACKAGE_DIR)/CIRC_npjDM_response_to_reviewers.pdf
	cp COVER_LETTER.tex RESPONSE_TO_REVIEWERS.tex RESPONSE_TO_REVIEWERS.md $(PACKAGE_DIR)/editorial/
	cp SUBMISSION_CHECKLIST.md $(PACKAGE_DIR)/README.md
	rm -f $(PACKAGE_ZIP)
	zip -qr $(PACKAGE_ZIP) $(PACKAGE_DIR)

clean:
	latexmk -C -outdir=build main.tex
	rm -f main.pdf $(FINAL_MANUSCRIPT_PDF) COVER_LETTER.pdf RESPONSE_TO_REVIEWERS.pdf
