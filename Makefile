.PHONY: all train predict league analyze render test provenance stage smoke clean
all: ; ./run_all.sh all
train: ; ./run_all.sh train
predict: ; ./run_all.sh predict
league: ; ./run_all.sh league
analyze: ; ./run_all.sh analyze
render: ; .venv/bin/python src/render.py
test: ; .venv/bin/python -m unittest discover -s tests -v
provenance: ; .venv/bin/python src/write_provenance.py
stage: ; .venv/bin/python src/build_release_stage.py --include-models --include-predictions --include-baselines
smoke: ; python3 ../others/smoke/smoke_test.py
clean: ; rm -rf results/preds results/*.csv results/summary.json
