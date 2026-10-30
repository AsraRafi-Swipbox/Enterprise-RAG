export PYTHONPATH := src
.PHONY: install test eval report diagram demo serve all
install:  ; pip install -r requirements.txt
test:     ; python -m pytest -q
eval:     ; python scripts/run_eval.py
report:   ; python scripts/make_report.py
diagram:  ; python scripts/make_diagram.py
demo:     ; python -m erag.cli demo
serve:    ; python -m erag.cli serve --port 8000   # needs ERAG_JWT_SECRET (>=16 chars)
all: test eval report diagram
