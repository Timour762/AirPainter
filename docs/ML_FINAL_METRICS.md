# ML Final Metrics (Sprint 4)

Date: 2026-03-16  
Branch: `feature/ml-sprints-roadmap`

## Dataset
- Source: `data/raw/landmarks_raw.csv` (bootstrap synthetic for pipeline validation).
- Size: 3000 samples total.
- Balance: 600 samples per class (`pause`, `draw`, `erase`, `clear`, `change_color`).
- Split:
  - train: 2100
  - val: 450
  - test: 450

## Baseline Training
- Script: `py scripts/train.py --train-csv data/processed/train.csv --val-csv data/processed/val.csv --output models/gesture_mlp.pt --epochs 25 --batch-size 64 --lr 0.001 --device cpu`
- Best validation accuracy: `1.0000`
- Output checkpoint: `models/gesture_mlp.pt`

## Test Evaluation
- Script: `py scripts/evaluate.py --checkpoint models/gesture_mlp.pt --test-csv data/processed/test.csv --device cpu`
- Accuracy: `1.0000`
- Confusion matrix:
  - `[[90,0,0,0,0],[0,90,0,0,0],[0,0,90,0,0],[0,0,0,90,0],[0,0,0,0,90]]`

## E2E Gesture Scenario (No Camera)
- Script: `py scripts/run_e2e_scenario.py --checkpoint models/gesture_mlp.pt --dataset data/raw/landmarks_clean.csv`
- Results:
  - draw adds pixels: PASS
  - erase reduces pixels: PASS
  - pause keeps pixels: PASS
  - clear resets canvas: PASS
  - change_color updates state: PASS
