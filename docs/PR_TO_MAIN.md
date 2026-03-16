# PR To Main (Prepared)

Date: 2026-03-16  
Target: `main`  
Source: `feature/ml-sprints-roadmap`

## PR Link
- Open PR: https://github.com/Timour762/AirPainter/pull/new/feature/ml-sprints-roadmap

## Scope
- Sprint 0 completed.
- Sprint 1 completed.
- Sprint 2 completed.
- Sprint 3 completed (including e2e action scenario).
- Sprint 4 completed (tests, smoke inference, docs, metrics).

## Validation Checklist
- [x] Unit tests: `py -m unittest discover -s tests -p "test_*.py" -v`
- [x] E2E scenario: `py scripts/run_e2e_scenario.py --checkpoint models/gesture_mlp.pt --dataset data/raw/landmarks_clean.csv`
- [x] Model eval: `py scripts/evaluate.py --checkpoint models/gesture_mlp.pt --test-csv data/processed/test.csv --device cpu`
- [x] Sprint board updated.

## Notes
- Real webcam live verification depends on camera availability on target machine.
- Current dataset is bootstrap synthetic and intended to validate pipeline/integration; for production behavior collect real camera landmarks with `scripts/collect_landmarks.py`.
