#!/usr/bin/env bash
# Create (or update) the private Kaggle dataset that gives Kaggle kernels the HF token.
# Run it yourself; the token is read from the environment and never written into the repo:
#   HF_TOKEN=hf_xxx KAGGLE_USER=<kaggle-user> bash scripts/kaggle_secret.sh
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN}" "${KAGGLE_USER:?set KAGGLE_USER}"
d=$(mktemp -d "${TMPDIR:-/var/tmp}/intentlock-secret.XXXXXX")
trap 'rm -rf "$d"' EXIT
printf '%s' "$HF_TOKEN" > "$d/hf_token.txt"
printf '{"title": "intentlock-secrets", "id": "%s/intentlock-secrets", "licenses": [{"name": "other"}]}\n' "$KAGGLE_USER" > "$d/dataset-metadata.json"
if kaggle datasets status "$KAGGLE_USER/intentlock-secrets" >/dev/null 2>&1; then
  kaggle datasets version -p "$d" -m "rotate token"
else
  kaggle datasets create -p "$d"          # private by default
fi
