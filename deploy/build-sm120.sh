#!/bin/bash
# Reproducible source build: DeepSeek-V4-Flash inference stack for SM120
# (2x NVIDIA RTX PRO 6000 Blackwell / RTX 50xx, no NVLink, CUDA 13.0).
#
# Builds into a dedicated conda env. Components and exact pins (mirrors
# local-inference-lab/blackwell-llm-docker build-lucifer-cu132.sh, adapted
# from Docker to a bare conda env):
#   vLLM       local-inference-lab/vllm @ lucifer (7c6bbf4)
#   FlashInfer flashinfer-ai/flashinfer @ PR #3395, commit b41aa8dd (SM120 sparse MLA)
#   DeepGEMM   deepseek-ai/DeepGEMM     @ PR #324,  commit aced12c2 (SM120 kernels)
# The PR heads are MOVING TARGETS (a later #3395 refactor renames the module
# vLLM imports) - always use the pinned commits below, never the PR heads.
#   CUTLASS DSL nvidia-cutlass-dsl 4.5.2 + blockscaled_layout.py from
#               NVIDIA/cutlass @ d80a4e53 (adds sm120 smem SF layouts)
#
# Prereqs: miniconda, CUDA toolkit 13.0 at /usr/local/cuda-13.0 (nvcc needed
# at build AND runtime - DeepGEMM/FlashInfer JIT-compile kernels), rustup
# (vLLM rust frontend), ~90GB RAM recommended for MAX_JOBS=16.
set -euo pipefail

ENV_NAME=deepseekv4
VLLM_SRC=~/vllm-lucifer
FLASHINFER_SRC=~/flashinfer-src
DEEPGEMM_SRC=~/deepgemm-src

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

source ~/miniconda3/etc/profile.d/conda.sh

# ── 1. Environment + pinned torch (cu130) ───────────────────────────────
conda create -y -n $ENV_NAME python=3.12
conda activate $ENV_NAME
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130
# CUDA 13.2 toolkit inside the env: DeepGEMM's SM120 JIT kernels fail
# template resolution under nvcc 13.0; runtime JIT must use this nvcc
# (start-deepseek.sh sets CUDA_HOME=$CONDA_PREFIX accordingly).
conda install -y -c nvidia cuda-toolkit=13.2

# ── 2. vLLM lucifer fork (this repo) ────────────────────────────────────
# ~1-1.5h on 32 threads. TORCH_CUDA_ARCH_LIST=12.0 -> cmake selects
# 12.0f/12.0a family targets per component (CUDA >= 13.0).
pip install -r "$VLLM_SRC/requirements/build/cuda.txt"
cd "$VLLM_SRC"
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=16 NVCC_THREADS=2 \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="0.24.0+lucifer.sm120" \
    pip install --no-build-isolation -v .

# ── 3. FlashInfer runtime deps + CUTLASS DSL ────────────────────────────
# WARNING: never install flashinfer-python from PyPI with deps enabled -
# its resolver will replace torch cu130 with an incompatible build.
pip install "apache-tvm-ffi>=0.1.6" "nvidia-cudnn-frontend>=1.13.0" \
    packaging pynvml nvidia-ml-py requests numpy einops click tabulate tqdm \
    "nvidia-cutlass-dsl[cu13]==4.5.2" nvidia-cutlass-dsl-libs-cu13==4.5.2

# ── 4. FlashInfer from source @ PR #3395 (SM120 sparse MLA wrapper) ─────
# Released wheels (<= 0.6.14) lack flashinfer.sparse_mla_sm120 /
# swa_indices / run_sparse_mla. Do NOT install flashinfer-cubin or
# flashinfer-jit-cache - prebuilt artifacts override the patched kernels.
if [ ! -d "$FLASHINFER_SRC" ]; then
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git "$FLASHINFER_SRC"
    git -C "$FLASHINFER_SRC" fetch origin refs/pull/3395/head
    git -C "$FLASHINFER_SRC" checkout b41aa8dd2fb93c49b1c6134bd1953040f8089d51
    git -C "$FLASHINFER_SRC" submodule update --init --recursive
fi
pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache || true
cd "$FLASHINFER_SRC"
FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_ENABLE_SM90=0 MAX_JOBS=16 \
    python -m pip install --no-build-isolation --no-deps -v .

# ── 5. Patch installed CUTLASS DSL with SM120 blockscaled layouts ───────
CUTLASS_DSL="$(python -c 'import cutlass.utils.blockscaled_layout as m; print(m.__file__)')"
curl -sf https://raw.githubusercontent.com/NVIDIA/cutlass/d80a4e53b52b42550659a8696dab32705265e324/python/CuTeDSL/cutlass/utils/blockscaled_layout.py \
    -o "$CUTLASS_DSL"
python -c 'from cutlass.utils.blockscaled_layout import sm120_make_smem_layout_sfa; print("DSL patch OK")'

# ── 6. DeepGEMM standalone @ PR #324 (SM120 support) ────────────────────
# vLLM prefers a site-packages deep_gemm over its vendored copy, so this
# supersedes the vendored build without recompiling vLLM.
if [ ! -d "$DEEPGEMM_SRC" ]; then
    git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git "$DEEPGEMM_SRC"
    git -C "$DEEPGEMM_SRC" fetch origin refs/pull/324/head
    git -C "$DEEPGEMM_SRC" checkout aced12c2c8882a945c568ace9d4a7e5778aae410
    git -C "$DEEPGEMM_SRC" submodule update --init --recursive
fi
cd "$DEEPGEMM_SRC"
rm -rf build  # stale objects from a different checkout cause template-arity JIT failures
DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=1 MAX_JOBS=16 \
    python -m pip install --no-build-isolation --no-deps --force-reinstall .

echo "Build complete. Launch with deploy/start-deepseek.sh"
