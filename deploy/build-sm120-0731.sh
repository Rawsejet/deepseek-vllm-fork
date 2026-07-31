#!/bin/bash
# Reproducible source build: DeepSeek-V4-Flash-0731 (DSpark refresh) for SM120
# (2x NVIDIA RTX PRO 6000 Blackwell / RTX 50xx, no NVLink, CUDA 13.0 + 13.2).
#
# The 0731 checkpoint adds DSpark block compression (config fields dspark_*,
# target layers 40-42) and DSpark block speculative decoding, so it needs a
# NEWER stack than the original V4-Flash. Builds into its own conda env; the
# original `deepseekv4` env is left intact. Pins mirror
# local-inference-lab/blackwell-llm-docker build-ds4dspark-v7-cu132.sh:
#   vLLM       local-inference-lab/vllm @ codex/ds4dspark-v7-pr60-20260628
#              commit ea45be31 (SPARSE_MLA alias dropped; DSpark spec decode in
#              the V2 model runner; adds humming MoE + b12x backends)
#   FlashInfer flashinfer-ai/flashinfer @ 25dd814e (sparse MLA under mla/, newer
#              layout than the original build's PR #3395 pin)
#   DeepGEMM   deepseek-ai/DeepGEMM @ nv_dev, commit 2073ddb2
#   CUTLASS DSL nvidia-cutlass-dsl 4.5.2 + blockscaled_layout.py from
#              NVIDIA/cutlass @ d80a4e53
#   humming-kernels[cu13]==0.1.6 ; torch 2.12.0+cu130 (up from 2.11 on original)
set -euo pipefail

ENV_NAME=deepseekv40731
VLLM_SRC=~/vllm-dspark
FLASHINFER_SRC=~/flashinfer-src-0731
DEEPGEMM_SRC=~/deepgemm-src-0731

# system CUDA 13.0 for the AOT compile; conda cuda-toolkit 13.2 for runtime JIT
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

source ~/miniconda3/etc/profile.d/conda.sh

# ── 1. Environment + pinned torch (cu130) + toolkits ────────────────────
conda create -y -n $ENV_NAME python=3.12
conda activate $ENV_NAME
pip install torch==2.12.0 torchvision==0.27.0 torchaudio==2.12.0 \
    --index-url https://download.pytorch.org/whl/cu130
conda install -y -c nvidia cuda-toolkit=13.2
# torch 2.12 cmake needs cuDNN dev headers present to configure (the docker
# build base is a cudnn-devel image; a bare env must supply them).
conda install -y -c nvidia cudnn

# ── 2. vLLM DSpark branch (~/vllm-dspark) ───────────────────────────────
# Uses system CUDA 13.0 for the AOT build. CRITICAL toolchain untangling:
# conda activation exports CMAKE_PREFIX_PATH / CMAKE_ARGS / LDFLAGS / CFLAGS
# that hijack cmake's CUDA + compiler discovery -> "Could NOT find CUDA".
# Unset them and shim system binutils ahead of conda's ld.
pip install -r "$VLLM_SRC/requirements/build/cuda.txt"
mkdir -p /tmp/sysbinutils
for t in ld ar ranlib nm strip objcopy objdump readelf as; do
    ln -sf /usr/bin/$t /tmp/sysbinutils/$t
done
cd "$VLLM_SRC" && rm -rf build .deps
env -u CMAKE_ARGS -u CMAKE_PREFIX_PATH -u LDFLAGS -u CFLAGS -u CXXFLAGS \
    -u CPPFLAGS -u DEBUG_CFLAGS -u DEBUG_CXXFLAGS \
    CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0 \
    PATH="/tmp/sysbinutils:/usr/local/cuda-13.0/bin:$PATH" \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++ \
    CUDNN_ROOT="$CONDA_PREFIX" CUDNN_INCLUDE_DIR="$CONDA_PREFIX/include" \
    CUDNN_LIBRARY="$CONDA_PREFIX/lib/libcudnn.so" \
    LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu" \
    TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=16 NVCC_THREADS=2 \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="0.11.2.dev279+ds4dspark.v7.sm120" \
    pip install --no-build-isolation -v .

# ── 3. FlashInfer deps + CUTLASS DSL + humming-kernels ──────────────────
pip install "apache-tvm-ffi>=0.1.6" "nvidia-cudnn-frontend>=1.13.0" \
    packaging pynvml nvidia-ml-py requests numpy einops click tabulate tqdm \
    "nvidia-cutlass-dsl[cu13]==4.5.2" nvidia-cutlass-dsl-libs-cu13==4.5.2 \
    "humming-kernels[cu13]==0.1.6"

# ── 4. FlashInfer from source @ 25dd814e (no cubin/jit-cache packages) ──
if [ ! -d "$FLASHINFER_SRC" ]; then
    git clone --recursive https://github.com/flashinfer-ai/flashinfer.git "$FLASHINFER_SRC"
    git -C "$FLASHINFER_SRC" fetch --depth=1 origin 25dd814e03791e370f96c3148242f0dc8de504ac
    git -C "$FLASHINFER_SRC" checkout 25dd814e03791e370f96c3148242f0dc8de504ac
    git -C "$FLASHINFER_SRC" submodule update --init --recursive
fi
pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache || true
cd "$FLASHINFER_SRC"
FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_ENABLE_SM90=0 MAX_JOBS=16 \
    python -m pip install --no-build-isolation --no-deps -v .

# ── 5. CUTLASS DSL SM120 blockscaled layout patch ───────────────────────
CUTLASS_DSL="$(python -c 'import cutlass.utils.blockscaled_layout as m; print(m.__file__)')"
curl -sf https://raw.githubusercontent.com/NVIDIA/cutlass/d80a4e53b52b42550659a8696dab32705265e324/python/CuTeDSL/cutlass/utils/blockscaled_layout.py \
    -o "$CUTLASS_DSL"
python -c 'from cutlass.utils.blockscaled_layout import sm120_make_smem_layout_sfa; print("DSL patch OK")'

# ── 6. DeepGEMM @ nv_dev 2073ddb2 (standalone; rm -rf build first) ──────
if [ ! -d "$DEEPGEMM_SRC" ]; then
    git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git "$DEEPGEMM_SRC"
    git -C "$DEEPGEMM_SRC" fetch origin nv_dev
    git -C "$DEEPGEMM_SRC" checkout 2073ddb2814892014c33ef4cd1c7d4c148baf1fe
    git -C "$DEEPGEMM_SRC" submodule update --init --recursive
fi
cd "$DEEPGEMM_SRC"
rm -rf build  # stale objects from a different checkout crash JIT at runtime
DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=1 MAX_JOBS=16 \
    python -m pip install --no-build-isolation --no-deps --force-reinstall .

echo "Build complete. Launch with deploy/start-deepseek.sh (choose model 2)."
