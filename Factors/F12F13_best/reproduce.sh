#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${PKG_DIR}/../.." && pwd)"

cd "$PROJECT_ROOT"

python "$PKG_DIR/src/reproduce_lb6627.py" \
  --historical-fold0 \
  "$PKG_DIR/reference/F12_fold0_historical_post_probs_long.csv" \
  --f12-leaderboard-dir \
  outputs/factor/F12_taxonomy_verifier/leaderboard \
  --f12-summary-dir \
  outputs/factor/F12_taxonomy_verifier \
  --f13-probs \
  outputs/factor/F13_evidence_taxonomy_verifier/leaderboard/F13_leaderboard_5fold_avg_probs_long.csv \
  --f13-summary-dir \
  outputs/factor/F13_evidence_taxonomy_verifier \
  --template \
  submissions/SatAndSeriousTokyoScript.csv \
  --reference \
  "$PKG_DIR/reference/SatAndSeriousTokyoScript_F12F13_w60_40_safe_LB6627.csv" \
  --output \
  "$PKG_DIR/results/SatAndSeriousTokyoScript_F12F13_w60_40_safe_REPRODUCED.csv"
