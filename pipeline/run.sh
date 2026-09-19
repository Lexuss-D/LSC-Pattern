#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=""
ARGS=()
while (($#)); do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [[ -n "$CONFIG" ]]; then
  set -a
  source "$CONFIG"
  set +a
fi
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$ROOT/.local/cache/matplotlib"
export HF_HOME="$ROOT/.local/cache/huggingface"
export TORCH_HOME="$ROOT/.local/cache/torch"
export XDG_CACHE_HOME="$ROOT/.local/cache"
export XDG_CONFIG_HOME="$ROOT/.local/config"
export VLLM_CONFIG_ROOT="$ROOT/.local/config/vllm"
export VLLM_CACHE_ROOT="$ROOT/.local/cache/vllm"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export FLASHINFER_WORKSPACE_BASE="$ROOT/.local/cache/flashinfer"
export TRITON_CACHE_DIR="$ROOT/.local/cache/triton"
export CUDA_CACHE_PATH="$ROOT/.local/cache/cuda"
export TMPDIR="$ROOT/.local/tmp"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
exec "${PYTHON:-python3}" "$ROOT/src/pipeline.py" "${ARGS[@]}"
