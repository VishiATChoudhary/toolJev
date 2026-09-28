#!/usr/bin/env bash
# Every hosted-Jev benchmark, sequential (the API times out under parallel load)
# and resumable: if credits run out, add credits and rerun this script.
#
#   TYPESAFE_API_KEY=... bash bench/hosted.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TYPESAFE_API_KEY:?set TYPESAFE_API_KEY}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
LIMIT="${LIMIT:-200}"  # positives and negatives per dataset

# 1. Routing + abstention: Jev judging fit (shipped default), and Jev reranking the top 15.
for router in tooljev-hosted tooljev-hosted-rerank; do
  uv run python -m bench.run --datasets when2call mcptoolbench livemcpbench \
    --routers "$router" --limit "$LIMIT" --resume
done
# 2. Confidence gating on 400 banking77 tickets.
uv run python -m bench.gating --backend hosted
# 3. Tables.
uv run python -m bench.report > bench/results/report.md
uv run python -m bench.report --match tooljev-hosted > bench/results/report_matched_hosted.md
