#!/bin/bash
# Start DeepSeek-V4-Flash Server (vLLM, conda env: vllm)
# Model: MoE 284B total / 13B active, FP8 + FP4 experts, 149GB on disk,
#        1M token max context. Requires BOTH GPUs (TP=2), ~75GB weights per GPU.
# Other models must NOT be running.
#
# vLLM 0.24.0 has native DeepseekV4 support: model, MTP speculative decoding,
# deepseek_v4 tool/reasoning parsers, and the V4 renderer/tokenizer (the model
# uses Python message encoding instead of a Jinja chat template - vLLM handles it).
#
# Runs on the "lucifer" vLLM fork (local-inference-lab/vllm, lucifer branch)
# built from source 2026-07-16 in conda env `deepseekv4` with sm_120a/12.0f
# kernels (source tree: ~/vllm-lucifer). Stock vLLM cannot run V4 on SM120
# (DeepGemm arch gap - vllm issues #47818/#48602); revisit stock once fixed.
# DeepGEMM JIT-compiles kernels at runtime and needs nvcc (CUDA 13.0 env
# sourced below). First launch takes ~5-6 min extra for JIT + CUDA graphs.

LOG_DIR=~/deepseek-service
MODEL_PATH=~/models/deepseek
LOG_FILE=$LOG_DIR/deepseek-v4-flash-vllm.log
PID_FILE=$LOG_DIR/deepseek-v4-server.pid
PORT_FILE=$LOG_DIR/deepseek-v4-server.port
BACKEND_FILE=$LOG_DIR/deepseek-v4-server.backend
DEFAULT_PORT=8095

# NCCL tuning for RTX PRO 6000 Blackwell (SM120, PCIe, no NVLink):
# P2P/CUMEM and P2P/IPC hang on SM120 - working transport is SHM/direct/direct
# NCCL_P2P_DISABLE=1 + NCCL_CUMEM_ENABLE=0 forces SHM transport which works
# --disable-custom-all-reduce required: vLLM's custom all-reduce IPC also hangs on SM120
# CUTE_DSL_ARCH: FlashInfer's cute-dsl JIT must target sm_120a on these GPUs.
# HF_HUB_OFFLINE: model is local; skip HF network calls at startup.
# VLLM_DEEP_GEMM_WARMUP=skip: the parallel warmup JIT spawns many concurrent
# nvcc/cicc compiles (~6GB RSS each) and OOM-killed the server on this 91GB
# box; lazy on-demand JIT compiles sequentially and caches to disk instead.
NCCL_ENV="NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_IB_DISABLE=1 CUTE_DSL_ARCH=sm_120a HF_HUB_OFFLINE=1 VLLM_DEEP_GEMM_WARMUP=skip VLLM_ENGINE_CORE_STARTUP_TIMEOUT=1800 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

wait_for_vllm_then_continue() {
    local PORT=$1
    local PID_FILE=$2
    local LOG_FILE=$3
    local START_TIME=$SECONDS
    local LAST_LINE=""
    echo "Waiting for server to be ready (149GB of weights - expect several minutes)..."
    echo "Press Ctrl+C to background — server will keep running."
    echo ""
    while true; do
        if ! kill -0 $(cat "$PID_FILE" 2>/dev/null) 2>/dev/null; then
            echo ""
            echo "✗ Server process died. Last 20 lines of log:"
            tail -20 "$LOG_FILE"
            exit 1
        fi
        if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            ELAPSED=$(( SECONDS - START_TIME ))
            echo ""
            echo "✓ Server is ready! (${ELAPSED}s)"
            echo ""
            return 0
        fi
        NEW_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null)
        if [ "$NEW_LINE" != "$LAST_LINE" ]; then
            ELAPSED=$(( SECONDS - START_TIME ))
            echo "  [${ELAPSED}s] $NEW_LINE"
            LAST_LINE="$NEW_LINE"
        fi
        sleep 2
    done
}

echo "========================================="
echo "  DeepSeek-V4-Flash Server"
echo "========================================="

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "DeepSeek-V4-Flash server already running with PID $PID"
        exit 1
    fi
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo ""
    echo "Model not found at $MODEL_PATH"
    echo "Download with:"
    echo "  hf download deepseek-ai/DeepSeek-V4-Flash --local-dir $MODEL_PATH"
    exit 1
fi

# Guard against a partially downloaded checkpoint (46 shards expected)
SHARD_TOTAL=$(ls "$MODEL_PATH"/model-*-of-*.safetensors 2>/dev/null | head -1 | grep -oP 'of-0*\K[0-9]+')
SHARD_COUNT=$(ls "$MODEL_PATH"/model-*-of-*.safetensors 2>/dev/null | wc -l)
if [ -z "$SHARD_TOTAL" ] || [ "$SHARD_COUNT" -lt "$SHARD_TOTAL" ]; then
    echo ""
    echo "Checkpoint incomplete: $SHARD_COUNT of ${SHARD_TOTAL:-?} shards present in $MODEL_PATH"
    echo "Is the download still running? Resume with:"
    echo "  hf download deepseek-ai/DeepSeek-V4-Flash --local-dir $MODEL_PATH"
    exit 1
fi

echo ""
read -p "Port [default: $DEFAULT_PORT]: " port_input
PORT=${port_input:-$DEFAULT_PORT}

# ─────────────────────────────────────────────
# Context size (always dual GPU, TP=2 - model does not fit on one GPU)
# ─────────────────────────────────────────────
echo ""
echo "Choose context size (dual GPU TP=2, ~75GB weights per GPU):"
echo ""
echo "1) 128K context (0.90 util)"
echo "   - Safe default, headroom for activations/CUDA graphs"
echo ""
echo "2) 256K context (0.92 util)"
echo "   - More room for long agentic sessions"
echo ""
echo "3) 384K context (0.95 util)"
echo "   - Minimum recommended for Think Max reasoning mode"
echo "   - Tight on VRAM; if startup OOMs, fall back to option 2"
echo ""
read -p "Enter choice [1-3]: " ctx_choice

case $ctx_choice in
    1)
        MEM_FRAC=0.90
        MAX_MODEL_LEN=131072
        CTX_LABEL="128K (0.90 util)"
        ;;
    2)
        MEM_FRAC=0.92
        MAX_MODEL_LEN=262144
        CTX_LABEL="256K (0.92 util)"
        ;;
    3)
        MEM_FRAC=0.95
        MAX_MODEL_LEN=393216
        CTX_LABEL="384K (0.95 util, Think Max)"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# ─────────────────────────────────────────────
# Mode
# ─────────────────────────────────────────────
echo ""
echo "Choose mode:"
echo ""
echo "1) Tools             tool calling + reasoning  (use with Claude Code)"
echo "2) Tools + MTP       tools + speculative decoding (faster decode)"
echo "3) Text only         reasoning only, no tool calling"
echo ""
read -p "Enter choice [1-3]: " mode_choice

case $mode_choice in
    1)
        MODE_FLAGS=(--reasoning-parser deepseek_v4 --enable-auto-tool-choice --tool-call-parser deepseek_v4)
        MODE_SPECULATIVE=""
        MODE_LABEL="Tools"
        ;;
    2)
        MODE_FLAGS=(--reasoning-parser deepseek_v4 --enable-auto-tool-choice --tool-call-parser deepseek_v4)
        echo ""
        echo "Choose MTP tokens (model ships 1 MTP layer):"
        echo "  1) 1 token  (safe default)"
        echo "  2) 2 tokens (layer reused, usually still faster)"
        read -p "Enter choice [1-2]: " mtp_choice
        case $mtp_choice in
            2) MODE_SPECULATIVE='{"method":"mtp","num_speculative_tokens":2}' ;;
            *) MODE_SPECULATIVE='{"method":"mtp","num_speculative_tokens":1}' ;;
        esac
        MODE_LABEL="Tools + MTP"
        ;;
    3)
        MODE_FLAGS=(--reasoning-parser deepseek_v4)
        MODE_SPECULATIVE=""
        MODE_LABEL="Text only"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Configuration:"
echo "  Backend:     vLLM lucifer fork, sm_120a build  (conda env: deepseekv4)"
echo "  Model:       DeepSeek-V4-Flash (MoE 284B/13B active, FP8+FP4, 149GB)"
echo "  GPUs:        Dual GPU TP=2 (required)"
echo "  Mode:        $MODE_LABEL"
echo "  Context:     $CTX_LABEL"
echo "  Port:        $PORT"
echo ""
echo "  Sampling (per model card): temperature=1.0, top_p=1.0"
echo "  Thinking is selected per request via chat_template_kwargs:"
echo "    {\"thinking\": true}   -> think mode"
echo "    omitted / false       -> non-think mode"
echo ""

SPEC_ARGS=()
if [ -n "$MODE_SPECULATIVE" ]; then
    SPEC_ARGS=("--speculative-config" "$MODE_SPECULATIVE")
fi

# Activate env directly instead of conda run to preserve argument quoting
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepseekv4

# DeepGEMM/FlashInfer JIT need nvcc at runtime. Must be CUDA 13.2 (env-local
# conda toolkit): nvcc 13.0 fails template resolution on the SM120
# fp8_fp4_gemm JIT kernels ("cannot determine which instance ... is intended").
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
# conda cuda-toolkit keeps headers under targets/, not $CUDA_HOME/include -
# FlashInfer's ninja JIT needs them on the compiler search path explicitly
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
# stubs dir provides libcuda.so for the -lcuda link step (conda's ld does
# not search system paths); runtime resolution still uses the real driver
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/targets/x86_64-linux/lib/stubs:/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"
# Cap runtime JIT compile parallelism: FlashInfer's ninja JIT defaults to all
# 32 cores and each nvcc/cicc instance peaks at ~5-6GB RSS -> OOM + desktop
# freeze on this 91GB box. 4 jobs x 2 TP workers stays well under RAM.
export MAX_JOBS=4
export NVCC_THREADS=1

# Flags per the lucifer fork's reference deployment on 2x RTX PRO 6000:
# --attention-backend SPARSE_MLA_SM120: fork's SM120 sparse MLA path
# --kernel-config.moe_backend flashinfer_cutlass: fork's adapted MXFP4 MoE
# --block-size 256 + --kv-cache-dtype fp8: required V4 cache layout
# --max-num-seqs 32: single-user box; keeps activation memory down on a model
# that already fills most of both GPUs with weights.
env $NCCL_ENV \
CUDA_VISIBLE_DEVICES=0,1 \
nohup vllm serve "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port $PORT \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization $MEM_FRAC \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs 32 \
    --served-model-name "DeepSeek-V4-Flash" \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --attention-backend SPARSE_MLA_SM120 \
    --kernel-config.moe_backend flashinfer_cutlass \
    --trust-remote-code \
    --disable-custom-all-reduce \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    "${MODE_FLAGS[@]}" \
    "${SPEC_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "$PORT" > "$PORT_FILE"
echo "vllm" > "$BACKEND_FILE"
echo "Server started with PID $(cat $PID_FILE)"
echo "Logs: $LOG_FILE"
echo ""
wait_for_vllm_then_continue $PORT $PID_FILE $LOG_FILE

# Warmup: pay the kernel-compilation cost now, not on the first real request
echo "Running warmup request (kernel compilation)..."
WARMUP_START=$SECONDS
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"DeepSeek-V4-Flash\",
        \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],
        \"max_tokens\": 5,
        \"temperature\": 1.0,
        \"top_p\": 1.0,
        \"chat_template_kwargs\": {\"thinking\": true}
    }" > /dev/null 2>&1
WARMUP_ELAPSED=$(( SECONDS - WARMUP_START ))
echo "✓ Warmup done (${WARMUP_ELAPSED}s). Server is ready for real requests."
echo ""
echo "Test with:"
echo "  curl http://127.0.0.1:$PORT/v1/models | python3 -m json.tool"
echo ""
