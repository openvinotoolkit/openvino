// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_gguf_opt.hpp"

#include <string>

#include "../primitive_ocl_base.hpp"
#include "../utils/fused_ops_jitter.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
#include "fully_connected_inst.h"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "openvino/core/type/element_type.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include <map>
#    include <memory>
#    include <mutex>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <oneapi/dnnl/dnnl_ocl.hpp>
#    include <unordered_map>
#    include <vector>

#    include "activation_inst.h"
#    include "eltwise_inst.h"
#    include "impls/onednn/utils.hpp"
#    include "intel_gpu/primitives/activation.hpp"
#    include "intel_gpu/primitives/eltwise.hpp"
#    include "intel_gpu/runtime/lru_cache.hpp"
#endif

namespace ov::intel_gpu::ocl {
namespace {

// JIT flag selecting the per-format block decoder (shared by the GEMV and transcode kernels).
const char* gguf_type_jit_flag(element::Type_t t) {
    switch (t) {
    case element::Type_t::gguf_q4_0:
        return "GGUF_IS_Q4_0";
    case element::Type_t::gguf_q4_1:
        return "GGUF_IS_Q4_1";
    case element::Type_t::gguf_q8_0:
        return "GGUF_IS_Q8_0";
    case element::Type_t::gguf_q4_k:
        return "GGUF_IS_Q4_K";
    case element::Type_t::gguf_q5_k:
        return "GGUF_IS_Q5_K";
    case element::Type_t::gguf_q6_k:
        return "GGUF_IS_Q6_K";
    case element::Type_t::gguf_q3_k:
        return "GGUF_IS_Q3_K";
    case element::Type_t::gguf_iq2_xs:
        return "GGUF_IS_IQ2_XS";
    case element::Type_t::gguf_iq2_s:
        return "GGUF_IS_IQ2_S";
    case element::Type_t::gguf_iq3_xxs:
        return "GGUF_IS_IQ3_XXS";
    case element::Type_t::gguf_iq3_s:
        return "GGUF_IS_IQ3_S";
    default:
        OPENVINO_THROW("[GPU] FCGGUFOpt: no kernel for GGUF element type ", element::Type(t).get_type_name());
    }
}

// Flattened activation rows BM = product of all activation dims except the last (K).
size_t derive_bm(const ov::Shape& shape_a) {
    size_t bm = 1;
    for (size_t i = 0; i + 1 < shape_a.size(); ++i) {
        bm *= shape_a[i];
    }
    return bm;
}

// fc_gguf_opt (float-decode GEMV) multi-row register-blocking factor: NROW output rows owned by one
// subgroup. For the M=1 decode GEMV every output row re-reads the SAME activation slice, so owning
// NROW rows amortises the per-row loop/decode overhead that throttles this scalar-float path. The win
// is shape-dependent (MAP-Elites-tuned on the B580): it helps moderate-N shapes that still have
// enough row-groups for occupancy, and HURTS huge-N shapes by cutting the row-group count. Tuned per
// (N, K): N4096-class shapes take NROW>1; the wide N>=12288 and tiny N<=1024 shapes keep NROW=1.
// Overridable via OV_GPU_GGUF_OPT_NROW for re-tuning on other hardware.
int gguf_opt_nrow(size_t N, size_t K) {
    if (const char* env = std::getenv("OV_GPU_GGUF_OPT_NROW")) {
        const long v = std::atol(env);
        if (v >= 1) {
            return static_cast<int>(v);
        }
    }
    // Verified back-to-back A/B on B580 (relL2 == baseline, layout unchanged):
    //   K12288 N4096 -> NROW=4 (x1.05),  K4096 N4096 -> NROW=8 (x1.06).
    // Wide-N (>=8192) and tiny-N (<=1024) shapes regress with NROW>1 -> keep 1.
    if (N >= 8192 || N <= 1024) {
        return 1;
    }
    if (N == 4096) {
        return (K >= 8192) ? 4 : 8;  // K12288 -> 4, K4096 -> 8
    }
    return 1;
}

// --------------------------------------------------------------------------------------------------
// GGUF -> OneDNN-WOQ transcode mapping (compute-bound path, SUMMARY §3.3.2 / SPEC §4.3).
// Each baseline GGUF format is requantised (ASYMMETRIC, per REQUANT_GROUP) into the smallest OneDNN
// low-bit unsigned weight domain that preserves its precision tier:
//   - 4-bit families (Q4_0, Q4_K)  -> u4 (unsigned 4-bit, [0..15] + u8 zero-point)
//   - IQ3_XXS                       -> u4 (sub-4 bit; see note below)
//   - 5/6-bit families              -> s8 (signed 8-bit, symmetric)
//   - Q8_0 ordinary FCs             -> s8; the wide LM head remains u8 asymmetric
//
// Asymmetric quantization matches the NNCF FP16-4BIT format (u4+u8 ZP+f16 scale) which oneDNN's
// jit:gemm:any W4A8 path natively supports on Xe2/B580, enabling DP4A utilization for GGUF prefill.
//
// IQ3 (XXS / S) rationale: every IQ3 element decodes to a sign times one of 8 codebook magnitudes
// per row (IQ3_XXS: {4, 12, 20, 28, 36, 44, 52, 62}; IQ3_S: {1, 3, 5, 7, 9, 11, 13, 15} from the
// 512-entry iq3s_grid). With sign that is 16 effective levels per group, which fits exactly into
// i4's [-8, 7]. Both formats use an ib32 sub-block aligned to REQUANT_GROUP=32, so each REQUANT
// group sees a single shared `db`; per-group symmetric requant to i4 then loses at most the
// smallest codebook magnitude when amax saturates the largest (ratio ~15.5:1 for IQ3_XXS,
// ~15:1 for IQ3_S), which is well below the ~3.0-3.4 bpw IQ3 quantisation noise itself. Mapping
// IQ3 to i8 instead would waste 7 bits/elem (only 16 of 256 levels carry signal) AND cost 2x
// weight bandwidth at the dnnl::matmul (see SUMMARY: i8 transcode for IQ3 was ~17 MiB/4Kx4K
// vs ~9 MiB at i4).
//
// IQ2_S rationale (i8, NOT i4): IQ2_S's per-element magnitudes come from a 1024-entry codebook
// whose row magnitudes are drawn from {8, 25, 43} -- 3 values, max/min ratio ~5.4:1. The catch is
// that each ib32 sub-block (= one REQUANT_GROUP of 32 elements) carries TWO independent 4-bit
// sub-scales -- db0 = d * (0.5 + scale[ib32]&0xF) * 0.25 for elements 0..15 and db1 = d * (0.5 +
// scale[ib32]>>4) * 0.25 for elements 16..31. Their ratio is bounded by (0.5+15)/(0.5+0) = 31:1,
// so the worst-case dynamic range within a single REQUANT_GROUP is ~5.4 * 31 ~= 167:1. i4 [-8,7]
// gives 1/7 ~= 14% min step, so any element below ~14% of the group amax would round to 0 -- a
// real risk when db0/db1 differ. i8 [-128,127] absorbs up to ~254:1 without signal loss, so we
// route IQ2_S through i8 (same tier as Q5_K / Q6_K / Q8_0). Bandwidth cost is small in practice:
// IQ2_S is a minority format in mixed-precision recipes (~10% of weight bytes in Qwen3-8B-UD).
//
// IQ2_XS rationale (i8, same tier as IQ2_S): IQ2_XS uses the same dual-sub-scale layout per ib32
// -- low/high nibble of scales[ib32] -> db0 (elems 0..15) / db1 (elems 16..31), each scaled by
// d*(0.5+nibble)*0.25 with ratio bound 31:1 -- and its 512-entry codebook draws from the same
// {8, 25, 43} magnitude alphabet as IQ2_S (max/min ~5.4:1). Worst-case in-group dynamic range is
// therefore the same ~167:1, well above i4's ~14:1 capacity. The only structural difference vs
// IQ2_S is *how* the 9-bit grid index + 7-bit sign-LUT index are packed (here: one 16-bit q per
// element-of-4 in each ib32, signs fetched via ksigns_iq2xs; IQ2_S splits qs/qh/signs across
// three parallel arrays). That packing affects the decoder math but not the requant headroom, so
// IQ2_XS routes to i8 by the same argument.
//
// Q3_K rationale (i4): Q3_K decodes each element as q in {-4..3} (2-bit payload plus a sign-like
// high-mask correction) multiplied by a per-16-element signed sub-scale `(sc - 32)` and global `d`.
// That is a 3-bit-centric quant tier with 8 logical levels per sub-block; mapping it to i4 keeps
// the same effective precision class while halving prefill weight bandwidth versus i8. Q3_K does
// not carry the IQ2-style dual-sub-scale-within-32 pathology (db0/db1 ratio up to 31:1), so the
// per-group symmetric requant noise at i4 is in-family for Q3-level models and materially below the
// cost of doubling bytes with i8.
//
// REQUANT_GROUP is fixed at 32 (divides every baseline block_elem: 32 and 256).
// --------------------------------------------------------------------------------------------------
constexpr int GGUF_REQUANT_GROUP = 32;  // fallback for block_elem=32 non-shuffle path

// Compute requant_group from block_elem.
// Default: 32, preserving the finest granularity for max precision.
// Override: OV_GPU_GGUF_REQUANT_GROUP can select another valid divisor of block_elem.
//   Must divide block_elem; values that don't divide are silently ignored (default used instead).
inline int gguf_requant_group(int block_elem) {
    if (const char* env = std::getenv("OV_GPU_GGUF_REQUANT_GROUP")) {
        const int v = std::atoi(env);
        if (v > 0 && block_elem % v == 0)
            return v;
    }
    return (block_elem >= 32) ? GGUF_REQUANT_GROUP : block_elem;
}

struct GgufTranscodeTarget {
    bool to_i4;       // true -> u4 (4-bit unsigned asymmetric), false -> s8 (8-bit signed symmetric)
    int qmax;         // value range max: 15 for u4 asymmetric, 127 for s8 symmetric
    bool asymmetric;  // true -> unsigned output + u8 ZP (u4 asym); false -> signed output, no ZP (s8 sym)
};

bool gguf_is_lm_head(size_t N) {
    return N >= 50000;
}

GgufTranscodeTarget transcode_target(element::Type_t t, size_t N) {
    switch (t) {
    case element::Type_t::gguf_q4_0:
    case element::Type_t::gguf_q4_1:
    case element::Type_t::gguf_q4_k:    // Q4_K also asymmetric u4, enabling jit:gemm:any W4A8
    case element::Type_t::gguf_q3_k:
    case element::Type_t::gguf_iq3_xxs:
    case element::Type_t::gguf_iq3_s:
        return {true,  15,  true};   // u4 asymmetric: [0..15] + u8 ZP
    case element::Type_t::gguf_q5_k:    // Q5_K/Q6_K: s8 symmetric, jit:gemm:any s8×s8 W4A8
    case element::Type_t::gguf_q6_k:
    case element::Type_t::gguf_iq2_xs:
    case element::Type_t::gguf_iq2_s:
        return {false, 127, false};  // s8 symmetric: [-127..127], no ZP
    case element::Type_t::gguf_q8_0:
        // Q8_0 is natively signed and was symmetrically transcoded before the NNCF-alignment
        // change. Keep ordinary FCs on s8/no-ZP so oneDNN can select its regular s8 GEMM. The
        // vocabulary projection is the exception: preserve its asymmetric WOQ path because its
        // very wide output is the workload for which the u8+ZP layout was introduced.
        return gguf_is_lm_head(N) ? GgufTranscodeTarget{false, 255, true}
                                  : GgufTranscodeTarget{false, 127, false};
    default:
        OPENVINO_THROW("[GPU] FCGGUFOpt: no transcode target for ", element::Type(t).get_type_name());
    }
}

// =================================================================================================
// Memory-bound (decode) GEMV kernel generator — unchanged native path, handles any M.
// =================================================================================================

// Subgroup width for the K-split GEMV: one subgroup (this many lanes) cooperatively computes one
// output, the reduction blocks of a row striped across its lanes. 16 matches the BMG/Xe2 native
// SIMD width and divides every shape's blocks-per-row (K is a multiple of 256 -> >= 16 blocks).
constexpr int GGUF_GEMV_SG_SIZE = 16;

#ifdef ENABLE_ONEDNN_FOR_GPU
// Q4_K / Q6_K "weight shuffle" path: when enabled, the weight was reordered once (compile_model) into
// the SG-transposed plane-separated layout that the sub-group-block-read GEMV kernels (fc_gguf_q4k_sg.cl
// / fc_gguf_q6k_sg.cl) and the shuffle-aware prefill transcode consume with fully coalesced weight
// loads. Same total bytes, bit-exact. Default ON; set OV_GPU_GGUF_SHUFFLE=0 to disable. Must match the
// RepackGGUFWeightsShuffle transform gate so the transform and the impl agree which nodes are shuffled.
bool gguf_shuffle_enabled() {
    if (const char* env = std::getenv("OV_GPU_GGUF_SHUFFLE")) {
        return std::atol(env) != 0;
    }
    return true;
}

// SG weight-shuffle format classification (shared by the transcode generator, the shuffle GEMV
// generators and the impl gate). Two families take the shuffle layout:
//   * K-block formats (Q4_K / Q5_K / Q6_K): native 256-elem block == one SG super-block.
//   * Small-block formats (Q4_0 / Q4_1 / Q8_0): native 32-elem block; EIGHT are grouped into one
//     256-elem super-block so they reuse the same OPG=16 sub-group-block-read machinery.
bool gguf_is_small_shuffle_format(element::Type_t t) {
    return t == element::Type_t::gguf_q4_0 || t == element::Type_t::gguf_q4_1 || t == element::Type_t::gguf_q8_0;
}
bool gguf_is_kblock_shuffle_format(element::Type_t t) {
    return t == element::Type_t::gguf_q4_k || t == element::Type_t::gguf_q5_k || t == element::Type_t::gguf_q6_k;
}
// Geometric + env gate. MUST match RepackGGUFWeightsShuffle's gate so the transform and the impl agree
// which nodes are shuffled: shuffle env on, a shuffle-eligible format, N % 16 == 0 (SG grouping) and K
// a whole number of 256-elem super-blocks.
bool gguf_shuffle_applicable(element::Type_t t, size_t N, size_t K) {
    if (!gguf_shuffle_enabled()) {
        return false;
    }
    if (!gguf_is_small_shuffle_format(t) && !gguf_is_kblock_shuffle_format(t)) {
        return false;
    }
    return (N % GGUF_GEMV_SG_SIZE == 0) && (K % 256 == 0);
}

// Opt-in f16 prefill (OV_GPU_GGUF_PREFILL_F16=1, default off). When on, the prefill transcode kernel
// fully dequantises the GGUF weight into an f16 scratchpad and the direct dnnl::matmul runs as a plain
// f16 x f16 GEMM (no WOQ scales) instead of the default i4/i8 weight-only-quantised path. This trades
// ~2-4x more weight bandwidth for reference-precision compute; useful for numeric comparison against
// the i8/i4 WOQ path. The transcode generator reads this directly (like gguf_shuffle_enabled) and the
// impl mirrors it into m_prefill_f16 so both sides agree.
bool gguf_prefill_f16_enabled() {
    if (const char* env = std::getenv("OV_GPU_GGUF_PREFILL_F16")) {
        return std::atol(env) != 0;
    }
    return false;
}

// DQ-activation W4A8 prefill (default: ON). The transcode kernel produces i4/i8 weight + scale
// (already), and now a new fc_gguf_act_dq kernel quantises the FP16 activation to INT8 + per-token
// f16 scale before the dnnl::matmul. Together this gives W4A8 DP4A on Xe hardware, matching the
// NNCF FP16-4BIT path. Set OV_GPU_GGUF_PREFILL_DQ_ACT=0 to fall back to WOQ-only (f16 activation).
bool gguf_prefill_dq_act_enabled() {
    if (const char* env = std::getenv("OV_GPU_GGUF_PREFILL_DQ_ACT")) {
        return std::atol(env) != 0;
    }
    return true;  // ON by default
}

// Weight-shuffle GEMV K-split (occupancy) heuristic (fc_gguf_q4k_sg.cl / fc_gguf_q6k_sg.cl).
// The shuffle GEMV launches one work-group per OPG (= GGUF_GEMV_SG_SIZE) output rows and, with
// KSPLIT>1, KSPLIT sub-groups per work-group, each covering a strided subset of the K-blocks. Total
// live hardware threads = (N/OPG) * KSPLIT. For small N the baseline (KSPLIT=1) leaves the GPU badly
// under-occupied; we raise KSPLIT so the total sub-group count approaches TARGET_THREADS, capped by
// the number of K-blocks (can't split more finely) and a work-group limit. Mirrors the reference
// choose_ksplit() in q4k_gemv/test_gemv_sg_kernels.py. Overridable via OV_GPU_GGUF_SHUFFLE_KSPLIT.
int gguf_shuffle_choose_ksplit(size_t N, size_t K) {
    // --- occupancy tuning knobs (env-overridable) ---
    const long TARGET_THREADS = 4096;   // threads we aim for
    const long BASE_KEEP      = 512;    // keep KSPLIT=1 above this many work-groups
    const long MAX_KSPLIT     = 16;     // work-group / block-count safety cap
    const long MIN_BLK_PER_SG = 2;      // keep >= this many K-blocks per sub-group
    const long MIN_THREADS    = 1024;   // ... unless still thread-starved
    const long OPG            = GGUF_GEMV_SG_SIZE;

    if (const char* env = std::getenv("OV_GPU_GGUF_SHUFFLE_KSPLIT")) {
        const long v = std::atol(env);
        if (v >= 1) {
            return static_cast<int>(v);
        }
    }

    const long nbpr = static_cast<long>(K) / 256;               // K-blocks available to split
    const long base = std::max<long>(1, static_cast<long>(N) / OPG);  // work-groups (== threads at KSPLIT=1)
    if (base >= BASE_KEEP) {                                    // already enough occupancy -> no split
        return 1;
    }

    // ks = pow2_floor(min(max(1, round(target/base)), nbpr, max_ksplit))
    const long want    = (TARGET_THREADS + base / 2) / base;   // round(target / base)
    long       capped  = std::max<long>(1, want);
    capped = std::min<long>(capped, nbpr);
    capped = std::min<long>(capped, MAX_KSPLIT);
    long ks = 1;
    while (ks * 2 <= capped) {
        ks *= 2;
    }

    // Back off if each sub-group would get too few K-blocks (reduction/SLM overhead dominates), but
    // only while we still have plenty of threads.
    while (ks > 1 && (nbpr / ks) < MIN_BLK_PER_SG && base * (ks / 2) >= MIN_THREADS) {
        ks /= 2;
    }
    return static_cast<int>(ks);
}

// -------------------------------------------------------------------------------------------------
// Fused post-op support for the prefill (transcode + dnnl::matmul) path.
//
// Decode (memory-bound GEMV) applies SwiGLU/residual fusions inside the OCL kernel via FUSED_OPS. The
// prefill path runs a dnnl::matmul, so the SAME fusions must be expressed as oneDNN post-ops, or the
// node would fall back to an unfused subgraph (no memory-pool reuse -> prefill activation blow-up).
//
// The fusions seen on GGUF FCs are: a unary `activation` (e.g. SwiGLU swish) and/or a binary
// `eltwise` (SwiGLU multiply by the up_proj output, or the residual add) whose second input is an
// EXTERNAL [M, N] tensor matching the matmul dst. Each maps 1:1 to a post-op:
//   activation(func, a, b) -> append_eltwise(convert_activation_func(func), a, b)
//   eltwise(prod|sum, ext) -> append_binary(binary_mul|binary_add, ext_md[M,N])
// EXTERNAL binary inputs are bound at execute via DNNL_ARG_ATTR_MULTIPLE_POST_OP(i)|DNNL_ARG_SRC_1.
struct GgufFusedOp {
    bool is_binary;  // false: unary eltwise(activation); true: binary(eltwise)
    dnnl::algorithm alg;
    float alpha = 0.0f;
    float beta = 0.0f;
    int ext_input_dep = -1;  // FC dependency index of the EXTERNAL binary src1 (or -1)
};

// Extract the ordered fused-op list from the FC's generic fused_desc. Returns empty if any fused op is
// not representable as a matmul post-op (caller then must not take the post-op path).
bool extract_gguf_fused_ops(const RuntimeParams& params, std::vector<GgufFusedOp>& out, bool& supported) {
    supported = true;
    out.clear();
    for (const auto& fd : params.fused_desc) {
        if (fd.is_type<cldnn::activation>()) {
            const auto desc = fd.typed_desc<cldnn::activation>();
            GgufFusedOp op;
            op.is_binary = false;
            op.alg = onednn::convert_activation_func(desc->activation_function);
            op.alpha = desc->additional_params.a;
            op.beta = desc->additional_params.b;
            out.push_back(op);
        } else if (fd.is_type<cldnn::eltwise>()) {
            const auto desc = fd.typed_desc<cldnn::eltwise>();
            GgufFusedOp op;
            op.is_binary = true;
            if (desc->mode == cldnn::eltwise_mode::prod) {
                op.alg = dnnl::algorithm::binary_mul;
            } else if (desc->mode == cldnn::eltwise_mode::sum) {
                op.alg = dnnl::algorithm::binary_add;
            } else if (desc->mode == cldnn::eltwise_mode::sub) {
                op.alg = dnnl::algorithm::binary_sub;
            } else {
                supported = false;
                return false;
            }
            // The binary peer is the single EXTERNAL dependency added after the FC's own inputs.
            op.ext_input_dep = fd.has_outer_dep() ? static_cast<int>(fd.outer_dep_start_idx) : -1;
            if (op.ext_input_dep < 0) {
                supported = false;
                return false;
            }
            out.push_back(op);
        } else if (fd.is_type<cldnn::reorder>()) {
            continue;  // reorder fusions are layout-only; ignored by the matmul path
        } else {
            supported = false;
            return false;
        }
    }
    return !out.empty();
}

// A compact, hashable signature of the fused-op chain for the matmul primitive cache key.
uint64_t gguf_fused_signature(const std::vector<GgufFusedOp>& ops) {
    uint64_t h = 1469598103934665603ull;
    const auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    for (const auto& op : ops) {
        mix(op.is_binary ? 2u : 1u);
        mix(static_cast<uint64_t>(op.alg));
        mix(static_cast<uint64_t>(op.ext_input_dep + 1));
    }
    return h;
}

// -------------------------------------------------------------------------------------------------
// Shared prefill transcode scratchpad.
//
// The prefill transcode produces a low-bit (u4/u8) weight + f16 scale + u8 zero-point that is
// consumed immediately by the very next dnnl::matmul enqueued on the SAME queue. oneDNN only supports
// an in-order queue (see ocl_stream::get_onednn_stream), so the transcode path always runs in-order:
// node L's matmul finishes reading the scratch before node L+1's transcode kernel starts (FIFO). A
// single scratch per stream, grown to the largest FC seen, can therefore be reused across every GGUF FC
// node instead of keeping one persistent transcoded copy per node.
struct TranscodeScratch {
    cldnn::memory::ptr weight;    // packed u4/u8 weight [N, K]         (asymmetric unsigned)
    cldnn::memory::ptr scale;     // f16 per-group scale  [K/group, N]
    cldnn::memory::ptr zp;        // u8  per-group zero-point [K/group, N]  (asymmetric, NEW)
    cldnn::memory::ptr act_int8;  // INT8 quantised activation [M_max, K]  (grow-only)
    cldnn::memory::ptr act_scale; // f16 per-token activation scale [M_max, 1]
};

struct TranscodeArena {
    std::mutex mtx;
    // Per-stream slot: each inference stream owns its own in-order queue, so its scratch must not be
    // shared with a concurrently-executing stream.
    std::unordered_map<const cldnn::stream*, TranscodeScratch> per_stream;
};

// Per-engine arena, owned by the live FC-GGUF impls (each holds a shared_ptr). The static registry
// only weakly references it, so the scratch is released — while the engine is still alive — as soon as
// the last GGUF FC instance of that engine is destroyed.
std::shared_ptr<TranscodeArena> get_transcode_arena(const cldnn::engine* eng) {
    static std::mutex g_mtx;
    static std::map<const cldnn::engine*, std::weak_ptr<TranscodeArena>> g_registry;
    std::lock_guard<std::mutex> lk(g_mtx);
    auto& weak = g_registry[eng];
    if (auto sp = weak.lock()) {
        return sp;
    }
    auto sp = std::make_shared<TranscodeArena>();
    weak = sp;
    return sp;
}

#endif  // ENABLE_ONEDNN_FOR_GPU

class FCGGUFOptGenerator : public KernelGenerator {
public:
    FCGGUFOptGenerator() : KernelGenerator("fc_gguf_opt") {}

protected:
    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override {
        const auto& in1 = params.input_layouts[1];
        const size_t N = in1.get_shape()[0];
        const size_t K = in1.get_shape()[1];
        std::string name = get_kernel_name() + "_" + element::Type(in1.data_type).get_type_name() + "_K" + std::to_string(K) + "_N" + std::to_string(N);
        const int nrow = gguf_opt_nrow(N, K);
        if (nrow != 1) {
            name += "_R" + std::to_string(nrow);  // NROW in the cache key so variants don't collide
        }
        if (params.is_dynamic()) {
            return name + "__sa";  // shape-agnostic (M from shape_info)
        }
        return name + "_M" + std::to_string(derive_bm(params.input_layouts[0].get_shape()));
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);
        jit.add(make_tensors_jit_constants(params));

        const auto& in0 = params.input_layouts[0];  // activation [BM, K]
        const auto& in1 = params.input_layouts[1];  // gguf weight [N, K] (always static)
        const auto& shape_w = in1.get_shape();

        jit.add(make_type_jit_constants("INPUT0", in0.data_type));
        jit.add(make_type_jit_constants("OUTPUT", params.output_layouts[0].data_type));

        const size_t N = shape_w[0];
        const size_t K = shape_w[1];
        const element::Type wt(in1.data_type);

        jit.add({
            make_jit_constant("K_SIZE", static_cast<int>(K)),
            make_jit_constant("N_SIZE", static_cast<int>(N)),
            make_jit_constant("GGUF_BLOCK_ELEM", static_cast<int>(wt.block_elem_count())),
            make_jit_constant("GGUF_BLOCK_BYTES", static_cast<int>(wt.block_byte_size())),
            make_jit_constant("SG_SIZE", GGUF_GEMV_SG_SIZE),
            make_jit_constant("NROW", gguf_opt_nrow(N, K)),
            make_jit_constant(gguf_type_jit_flag(in1.data_type), 1),
        });

        // SwiGLU / residual eltwise + activation fused onto this FC apply at output-write time, so the
        // dequantized GEMV result `dequantized` is post-processed in registers (no separate kernels,
        // no unfused subgraph, output stays in the reusable memory pool). The idx order matches
        // gguf_output_index(): BM -> (out_b, out_f), n -> channel(y) axis.
        if (params.has_fused_primitives()) {
            const auto& out_l = params.output_layouts[0];
            const std::vector<std::string> idx_order = {"out_b", "out_f", "n", "0"};
            FusedOpsConfiguration conf = {"", idx_order, "dequantized", out_l.data_type};
            jit.add(make_fused_ops_jit_constants(params, {conf}));
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        add_fused_ops_arguments(args, params);
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto& in1 = params.get_input_layout(1);
            const size_t N = in1.get_shape()[0];
            const size_t K = in1.get_shape()[1];
            const size_t BM = derive_bm(params.get_input_layout(0).get_shape());

            // One subgroup (GGUF_GEMV_SG_SIZE lanes) owns NROW output rows (multi-row register
            // blocking). global[0] = ceil(N/NROW) * SG_SIZE, local[0] = SG_SIZE keeps one subgroup
            // per work-group. NROW=1 reproduces the original one-subgroup-per-row geometry (max
            // work-groups -> best occupancy, critical for the small-N k/v projections). The dispatch
            // NROW MUST match the JIT NROW (gguf_opt_nrow) so the grid covers exactly ceil(N/NROW).
            const size_t nrow = static_cast<size_t>(gguf_opt_nrow(N, K));
            const size_t row_groups = (N + nrow - 1) / nrow;
            auto& wgs = kd.params.workGroups;
            wgs.global = {row_groups * GGUF_GEMV_SG_SIZE, BM, 1};
            wgs.local = {GGUF_GEMV_SG_SIZE, 1, 1};
        }};
    }
};

#ifdef ENABLE_ONEDNN_FOR_GPU
// =================================================================================================
// Transcode kernel generator (GGUF block -> i4/i8 weight + f16 per-group scale scratchpad).
// Shape-independent of M: keyed by the static weight only. Args are bound explicitly by the impl,
// so get_arguments_desc returns an empty descriptor (filled at dispatch time).
// =================================================================================================
class FCGGUFTranscodeGenerator : public KernelGenerator {
public:
    FCGGUFTranscodeGenerator() : KernelGenerator("fc_gguf_transcode") {}

protected:
    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override {
        const auto& in1 = params.input_layouts[1];
        const size_t N = in1.get_shape()[0];
        const size_t K = in1.get_shape()[1];
        return get_kernel_name() + "_" + element::Type(in1.data_type).get_type_name() + "_K" + std::to_string(K) + "_N" + std::to_string(N);
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);

        const auto& in1 = params.input_layouts[1];
        const auto& shape_w = in1.get_shape();
        const size_t N = shape_w[0];
        const size_t K = shape_w[1];
        const element::Type wt(in1.data_type);
        const auto tgt = transcode_target(in1.data_type, N);

        // Q4_K / Q5_K / Q6_K weight-shuffle path: the weight input to transcode is the SG-shuffled
        // buffer. Shuffled Q4_K/Q5_K/Q6_K now use the same requant path as non-shuffled formats:
        // GGUF_SHUFFLE=1 enables coalesced reads of the shuffled layout; TRANSCODE_TO_F16=0 so the
        // decoded blk_vals go through asymmetric (Q4_K: u4) or symmetric (Q5_K/Q6_K: s8) requant.
        const bool shuffled = gguf_shuffle_applicable(in1.data_type, N, K);
        // Small-block formats (Q4_0/Q4_1/Q8_0) shuffle 8 native 32-elem blocks into one 256-elem
        // super-block, so the transcode work-item owns a 256-elem super-block (GGUF_BLOCK_ELEM=256,
        // GGUF_BLOCK_BYTES = 8 * native bytes). K-block formats keep their native 256-elem block.
        const bool small_shuffle = shuffled && gguf_is_small_shuffle_format(in1.data_type);
        const int block_elem  = small_shuffle ? 256 : static_cast<int>(wt.block_elem_count());
        const int block_bytes = small_shuffle ? static_cast<int>(wt.block_byte_size() * 8) : static_cast<int>(wt.block_byte_size());
        // TRANSCODE_TO_F16 only when OV_GPU_GGUF_PREFILL_F16=1 (debug/comparison opt-in).
        // K-block shuffles no longer force f16: they produce u4 (Q4_K) or s8 (Q5_K/Q6_K) via requant.
        const bool to_f16 = gguf_prefill_f16_enabled();
        // requant_group=128 matches NNCF FP16-4BIT (4× fewer scale/ZP loads in the matmul vs group=32).
        // block_elem=256 (shuffled or K-block) allows group=128; block_elem=32 stays at 32.
        const int requant_group = gguf_requant_group(block_elem);
        jit.add({
            make_jit_constant("K_SIZE", static_cast<int>(K)),
            make_jit_constant("N_SIZE", static_cast<int>(N)),
            make_jit_constant("GGUF_BLOCK_ELEM", block_elem),
            make_jit_constant("GGUF_BLOCK_BYTES", block_bytes),
            make_jit_constant("REQUANT_GROUP", requant_group),
            make_jit_constant("TRANSCODE_TO_I4", tgt.to_i4 ? 1 : 0),
            make_jit_constant("QMAX", tgt.qmax),              // 15 (u4 asymmetric) or 127 (s8 symmetric)
            make_jit_constant("TRANSCODE_ASYMMETRIC", tgt.asymmetric ? 1 : 0),  // 0 -> s8 symmetric path
            make_jit_constant("GGUF_SHUFFLE", shuffled ? 1 : 0),
            make_jit_constant("TRANSCODE_TO_F16", to_f16 ? 1 : 0),
            make_jit_constant(gguf_type_jit_flag(in1.data_type), 1),
        });
        return jit;
    }

    // Args (raw weight in; packed weight + scale out) are supplied explicitly by the impl at dispatch.
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams&) const override {
        return {};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {}};
    }
};

// =================================================================================================
// Q4_K / Q5_K / Q6_K weight-shuffle GEMV generators (fc_gguf_q4k_sg.cl / fc_gguf_q5k_sg.cl /
// fc_gguf_q6k_sg.cl). Read the
// SG-transposed plane-separated weight with intel_sub_group_block_read (fully coalesced weight loads).
// Each lane owns ONE output row (no cross-lane reduce). SG_SIZE = 16 = OPG = the row-group size.
// Activation is f16 (INPUT0). Explicit-args; static K, N + output dtype baked in.
// =================================================================================================
class FCGGUFShuffleGenerator : public KernelGenerator {
public:
    explicit FCGGUFShuffleGenerator(const char* kernel_name) : KernelGenerator(kernel_name) {}

protected:
    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override {
        const auto& in1 = params.input_layouts[1];
        return get_kernel_name() + "_K" + std::to_string(in1.get_shape()[1]) + "_N" + std::to_string(in1.get_shape()[0]);
    }
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);
        const auto& in0 = params.input_layouts[0];  // activation [BM, K]
        const auto& in1 = params.input_layouts[1];  // gguf weight [N, K]
        const size_t N = in1.get_shape()[0];
        const size_t K = in1.get_shape()[1];
        jit.add(make_tensors_jit_constants(params));
        jit.add(make_type_jit_constants("INPUT0", in0.data_type));
        jit.add(make_type_jit_constants("OUTPUT", params.output_layouts[0].data_type));
        jit.add({
            make_jit_constant("K_SIZE", static_cast<int>(K)),
            make_jit_constant("N_SIZE", static_cast<int>(N)),
            make_jit_constant("SG_SIZE", GGUF_GEMV_SG_SIZE),
            make_jit_constant("OPG", GGUF_GEMV_SG_SIZE),
            // K-split (occupancy) factor: KSPLIT sub-groups cooperate on one row-group over disjoint
            // K-slices, reduced in SLM. MUST match the runtime dispatch geometry in execute_shuffle_gemv.
            make_jit_constant("KSPLIT", gguf_shuffle_choose_ksplit(N, K)),
        });
        // Same SwiGLU/residual fused-op handling as the other decode paths: applied at output-write.
        if (params.has_fused_primitives()) {
            const auto& out_l = params.output_layouts[0];
            const std::vector<std::string> idx_order = {"out_b", "out_f", "n", "0"};
            FusedOpsConfiguration conf = {"", idx_order, "dequantized", out_l.data_type};
            jit.add(make_fused_ops_jit_constants(params, {conf}));
        }
        return jit;
    }
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams&) const override {
        return {};
    }
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {}};
    }
};

class FCGGUFQ4kSgGenerator : public FCGGUFShuffleGenerator {
public:
    FCGGUFQ4kSgGenerator() : FCGGUFShuffleGenerator("fc_gguf_q4k_sg") {}
};

class FCGGUFQ5kSgGenerator : public FCGGUFShuffleGenerator {
public:
    FCGGUFQ5kSgGenerator() : FCGGUFShuffleGenerator("fc_gguf_q5k_sg") {}
};

class FCGGUFQ6kSgGenerator : public FCGGUFShuffleGenerator {
public:
    FCGGUFQ6kSgGenerator() : FCGGUFShuffleGenerator("fc_gguf_q6k_sg") {}
};

// Small-block (Q4_0 / Q4_1 / Q8_0) shuffle GEMV generators. Same coalesced sub-group-block-read GEMV
// over the SG-shuffled super-block weight; each owns its own kernel entry point.
class FCGGUFQ40SgGenerator : public FCGGUFShuffleGenerator {
public:
    FCGGUFQ40SgGenerator() : FCGGUFShuffleGenerator("fc_gguf_q4_0_sg") {}
};

class FCGGUFQ41SgGenerator : public FCGGUFShuffleGenerator {
public:
    FCGGUFQ41SgGenerator() : FCGGUFShuffleGenerator("fc_gguf_q4_1_sg") {}
};

class FCGGUFQ80SgGenerator : public FCGGUFShuffleGenerator {
public:
    FCGGUFQ80SgGenerator() : FCGGUFShuffleGenerator("fc_gguf_q8_0_sg") {}
};

// =================================================================================================
// Per-token INT8 activation DQ generator (fc_gguf_act_dq.cl).
// Quantises FP16 activation [M, K] -> INT8 [M, K] + f16 scale [M, 1].
// One work-group per token; WG size and K baked as JIT constants.
// =================================================================================================
constexpr int GGUF_ACT_DQ_WG_SIZE = 256;

class FCGGUFActDQGenerator : public KernelGenerator {
public:
    FCGGUFActDQGenerator() : KernelGenerator("fc_gguf_act_dq") {}

protected:
    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override {
        const size_t K = params.input_layouts[1].get_shape()[1];
        return get_kernel_name() + "_K" + std::to_string(K);
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);
        const size_t K = params.input_layouts[1].get_shape()[1];
        jit.add({
            make_jit_constant("K_SIZE",         static_cast<int>(K)),
            make_jit_constant("ACT_DQ_WG_SIZE", GGUF_ACT_DQ_WG_SIZE),
        });
        return jit;
    }

    // Args are supplied explicitly by execute_transcode_plus_onednn_woq.
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams&) const override { return {}; }
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {}};
    }
};
#endif  // ENABLE_ONEDNN_FOR_GPU

class FCGGUFOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::FCGGUFOptImpl)

    // Memory-bound GEMV stage (decode / small M) — always present.
    Stage::Ptr gguf_stage = make_stage<FCGGUFOptGenerator>();
#ifdef ENABLE_ONEDNN_FOR_GPU
    // Compute-bound transcode stage (prefill / large M) — feeds a direct dnnl::matmul.
    Stage::Ptr transcode_stage = make_stage<FCGGUFTranscodeGenerator>();
    // Per-token activation DQ stage (prefill W4A8 path): FP16 -> INT8 + scale.
    Stage::Ptr act_dq_stage = make_stage<FCGGUFActDQGenerator>();
    // Q4_K / Q6_K weight-shuffle decode path (OV_GPU_GGUF_SHUFFLE=1): coalesced sub-group-block-read
    // GEMV over the SG-shuffled weight that RepackGGUFWeightsShuffle produced at compile_model. One
    // stage per format (each owns its own kernel entry point).
    Stage::Ptr q4k_sg_stage = make_stage<FCGGUFQ4kSgGenerator>();
    Stage::Ptr q5k_sg_stage = make_stage<FCGGUFQ5kSgGenerator>();
    Stage::Ptr q6k_sg_stage = make_stage<FCGGUFQ6kSgGenerator>();
    // Small-block (Q4_0 / Q4_1 / Q8_0) weight-shuffle decode stages.
    Stage::Ptr q4_0_sg_stage = make_stage<FCGGUFQ40SgGenerator>();
    Stage::Ptr q4_1_sg_stage = make_stage<FCGGUFQ41SgGenerator>();
    Stage::Ptr q8_0_sg_stage = make_stage<FCGGUFQ80SgGenerator>();
#endif

    // Activation rows above which the transcode + OneDNN WOQ GEMM path is used (SUMMARY §5,
    // M_MEM_BOUND_THRESHOLD). Overridable via OV_GPU_GGUF_PREFILL_THRESHOLD for tuning.
    size_t m_prefill_threshold = 32;

    // Q4_K / Q5_K / Q6_K weight-shuffle decode path (OV_GPU_GGUF_SHUFFLE=1): the weight was reordered into
    // the SG-transposed plane-separated layout at compile_model (RepackGGUFWeightsShuffle replaced the
    // Constant), so the sub-group-block-read GEMV reads it fully coalesced. m_q4k_shuffle / m_q5k_shuffle /
    // m_q6k_shuffle record whether this node took the path. When set, decode uses the SG GEMV and prefill
    // uses the shuffle-aware transcode + f16 matmul.
    bool m_q4k_shuffle = false;
    bool m_q5k_shuffle = false;
    bool m_q6k_shuffle = false;
    // Small-block (Q4_0 / Q4_1 / Q8_0) weight-shuffle decode path (same gate as the K-block formats):
    // decode uses the coalesced SG GEMV, prefill uses the shuffle-aware transcode + f16 matmul.
    bool m_q4_0_shuffle = false;
    bool m_q4_1_shuffle = false;
    bool m_q8_0_shuffle = false;

    // Opt-in f16 prefill (OV_GPU_GGUF_PREFILL_F16=1, default off): the prefill transcode fully
    // dequantises the GGUF weight to an f16 scratchpad and the direct dnnl::matmul runs as a plain
    // f16 x f16 GEMM (no WOQ scales) instead of the i4/i8 weight-only-quantised path. Mirrors
    // gguf_prefill_f16_enabled() so the impl and the transcode generator agree.
    bool m_prefill_f16 = false;
    // DQ-activation W4A8 prefill: per-token INT8 activation quantisation + W4A8 DP4A dnnl::matmul.
    // Enabled by default (OV_GPU_GGUF_PREFILL_DQ_ACT=1); overridden to false when m_prefill_f16=true.
    bool m_use_dq_act = false;

#ifdef ENABLE_ONEDNN_FOR_GPU
    // Shared prefill transcode scratchpad for this engine (see TranscodeArena). Lazily fetched on the
    // first prefill execute; reused across all GGUF FC nodes so only one transcoded weight copy is
    // resident per stream instead of one persistent copy per node.
    std::shared_ptr<TranscodeArena> m_transcode_arena;
#endif

    FCGGUFOptImpl() : PrimitiveImplOCL(FCGGUFOpt::get_type_info_static()) {
        if (const char* env = std::getenv("OV_GPU_GGUF_PREFILL_THRESHOLD")) {
            const long v = std::atol(env);
            if (v >= 0) {
                m_prefill_threshold = static_cast<size_t>(v);
            }
        }
        if (const char* env = std::getenv("OV_GPU_GGUF_PREFILL_F16")) {
            m_prefill_f16 = (std::atol(env) != 0);
        }
        m_use_dq_act = !m_prefill_f16 && gguf_prefill_dq_act_enabled();
    }
    FCGGUFOptImpl(const program_node& node, const RuntimeParams& params) : FCGGUFOptImpl() {
        add_stage(gguf_stage, params);
#ifdef ENABLE_ONEDNN_FOR_GPU
        add_stage(transcode_stage, params);
        const auto wet = params.input_layouts[1].data_type;

        // Weight-shuffle gate for Q4_K / Q5_K / Q6_K (native 256-elem block) and the small-block formats
        // Q4_0 / Q4_1 / Q8_0 (native 32-elem block, grouped 8-to-a super-block): the RepackGGUFWeightsShuffle
        // transform reordered the weight into the SG layout iff shuffle env on, static weight, N % 16 == 0
        // and K % 256 == 0. gguf_shuffle_applicable() shares the same formula so the transform and impl
        // stay in lockstep. When shuffled, decode uses the SG GEMV and prefill uses the shuffle-aware
        // transcode + f16 matmul.
        const bool shuffle_applicable = [&]() {
            if (params.input_layouts[1].is_dynamic()) {
                return false;
            }
            const auto& wl = params.input_layouts[1];
            const size_t N = wl.get_shape()[0];
            const size_t K = wl.get_shape()[1];
            return gguf_shuffle_applicable(wl.data_type, N, K);
        }();

        if (shuffle_applicable && wet == element::Type_t::gguf_q4_k) {
            m_q4k_shuffle = true;
            add_stage(q4k_sg_stage, params);
        } else if (shuffle_applicable && wet == element::Type_t::gguf_q5_k) {
            m_q5k_shuffle = true;
            add_stage(q5k_sg_stage, params);
        } else if (shuffle_applicable && wet == element::Type_t::gguf_q6_k) {
            m_q6k_shuffle = true;
            add_stage(q6k_sg_stage, params);
        } else if (shuffle_applicable && wet == element::Type_t::gguf_q4_0) {
            m_q4_0_shuffle = true;
            add_stage(q4_0_sg_stage, params);
        } else if (shuffle_applicable && wet == element::Type_t::gguf_q4_1) {
            m_q4_1_shuffle = true;
            add_stage(q4_1_sg_stage, params);
        } else if (shuffle_applicable && wet == element::Type_t::gguf_q8_0) {
            m_q8_0_shuffle = true;
            add_stage(q8_0_sg_stage, params);
        }
        // DQ activation stage: always added when oneDNN is enabled; only executed during prefill.
        add_stage(act_dq_stage, params);
#endif
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        auto copy = make_deep_copy<FCGGUFOptImpl>(this);
        auto* c = static_cast<FCGGUFOptImpl*>(copy.get());
        c->m_prefill_threshold = m_prefill_threshold;
        c->m_q4k_shuffle = m_q4k_shuffle;
        c->m_q5k_shuffle = m_q5k_shuffle;
        c->m_q6k_shuffle = m_q6k_shuffle;
        c->m_q4_0_shuffle = m_q4_0_shuffle;
        c->m_q4_1_shuffle = m_q4_1_shuffle;
        c->m_q8_0_shuffle = m_q8_0_shuffle;
        c->m_prefill_f16 = m_prefill_f16;
        c->m_use_dq_act   = m_use_dq_act;
        return copy;
    }

    // Bind activation (INPUT0) + weight (INPUT1) + output for the GEMV stage. The empty scale/ZP FC
    // dependencies are intentionally not referenced by that kernel's descriptor.
    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        cldnn::kernel_arguments_data data;
        const auto* fc_inst = dynamic_cast<const cldnn::fully_connected_inst*>(&instance);

        data.inputs.push_back(instance.dep_memory_ptr(0));  // activation
        if (fc_inst) {
            data.inputs.push_back(fc_inst->weights_memory());
        } else {
            data.inputs.push_back(instance.dep_memory_ptr(1));
        }

        // Fused SwiGLU/residual eltwise inputs (e.g. the up_proj output or the residual tensor) are bound
        // as INPUT_OF_FUSED_PRIMITIVE args for the GEMV stage's FUSED_OPS code (see FCGGUFOptGenerator).
        if (instance.has_fused_primitives()) {
            const size_t count = instance.get_fused_mem_count();
            for (size_t i = 0; i < count; ++i) {
                data.fused_op_inputs.push_back(instance.fused_memory(i));
            }
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); ++i) {
            data.outputs.push_back(instance.output_memory_ptr(i));
        }
        data.shape_info = instance.shape_info_memory_ptr();
        return data;
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        // Refresh per-stage need_args_update / need_dispatch_data_update from the current execution
        // flags (SHAPE_CHANGED, ARG_UPDATE_REQUIRED, ...). The base execute() does this; since we
        // override execute() and dispatch the GEMV stage directly, we must do it too. Without it the
        // shape-agnostic GEMV kernel keeps the global_work_size computed for the first (prefill) shape
        // and re-runs decode (M=1) with the prefill row count, writing past the M=1 output buffer
        // (CL_OUT_OF_RESOURCES / out-of-bounds).
        update_rt_params(instance);
        // Dynamic decode may execute consecutive tokens with the same concrete shape while the memory
        // objects backing the activation/output change. Rebind arguments every time so the GEMV stage
        // does not read the previous token's activation when no SHAPE_CHANGED flag is raised.
        gguf_stage->kd.need_args_update = true;
        // The GEMV stage dispatch depends on the concrete runtime activation/output shape. Consecutive
        // decode iterations often have the same rank but different runtime buffers and may not carry a
        // SHAPE_CHANGED flag all the way to this custom multi-stage execute() path, so refresh dispatch
        // unconditionally before enqueueing the shape-agnostic kernel.
        gguf_stage->kd.need_dispatch_data_update = true;
#ifdef ENABLE_ONEDNN_FOR_GPU
        const auto& params = *instance.get_impl_params();
        const auto& in0 = params.get_input_layout(0);
        const auto& in1 = params.get_input_layout(1);
        if (!in0.is_dynamic() && !in1.is_dynamic()) {
            const size_t M = derive_bm(in0.get_shape());
            // Any M at or below the prefill threshold runs on the memory-bound decode path
            // (plain GEMV, or for shuffle-eligible formats the coalesced SG GEMV); above it,
            // the compute-bound transcode + oneDNN prefill path takes over.
            const size_t decode_cutoff = m_prefill_threshold;
            if (M > decode_cutoff) {
                // Prefill: Q4_K/Q6_K shuffle nodes run the shuffle-aware transcode -> f16 matmul; all
                // other formats run the WOQ (i4/i8) transcode + oneDNN matmul.
                return execute_transcode_plus_onednn_woq(events, instance, M);
            }
            // Decode: Q4_K/Q5_K/Q6_K and small-block Q4_0/Q4_1/Q8_0 shuffle nodes run the coalesced
            // sub-group-block-read GEMV.
            if (m_q4k_shuffle || m_q5k_shuffle || m_q6k_shuffle || m_q4_0_shuffle || m_q4_1_shuffle || m_q8_0_shuffle) {
                return execute_shuffle_gemv(events, instance, M);
            }
        }
#endif
        // Memory-bound / small-M (or OneDNN disabled): run only the GEMV stage.
        return execute_stage(events, instance, gguf_stage);
    }

#ifdef ENABLE_ONEDNN_FOR_GPU
private:
    // Bring the base auto-args execute_stage overloads into scope (the explicit-args overload below
    // would otherwise hide them by name).
    using PrimitiveImplOCL::execute_stage;

    // Explicit-args stage dispatch (the base execute_stage binds args via get_arguments(), which only
    // knows the FC deps). The transcode kernel reads the raw weight and writes the two scratchpads,
    // so its inputs/outputs/gws are supplied directly here (mirrors moe_3gemm_swiglu_opt's helper).
    cldnn::event::ptr execute_stage(const std::vector<cldnn::event::ptr>& events,
                                    cldnn::primitive_inst& instance,
                                    Stage& stage,
                                    std::vector<cldnn::memory::ptr> inputs,
                                    std::vector<cldnn::memory::ptr> outputs,
                                    const std::vector<size_t>& global,
                                    const std::vector<size_t>& local,
                                    const std::vector<cldnn::memory::ptr>& fused_inputs = {},
                                    bool needs_shape_info = false) const {
        cldnn::stream& stream = instance.get_network().get_stream();
        cldnn::kernel_arguments_data args;
        cldnn::kernel_arguments_desc desc;
        // Dynamic-shape kernels that carry fused ops index the external eltwise input via shape_info
        // (OPTIONAL_SHAPE_INFO_ARG). It must be the first argument, matching the kernel signature.
        if (needs_shape_info) {
            desc.arguments.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
            args.shape_info = instance.shape_info_memory_ptr();
        }
        for (uint32_t i = 0; i < inputs.size(); i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, i});
            args.inputs.push_back(inputs[i]);
        }
        for (uint32_t i = 0; i < outputs.size(); i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, i});
            args.outputs.push_back(outputs[i]);
        }
        // Fused-op (SwiGLU/residual eltwise) external inputs, bound to the FUSED_OPS_DECLS pointers in
        // the same order the generator appended them via add_fused_ops_arguments().
        for (uint32_t i = 0; i < fused_inputs.size(); i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, i});
            args.fused_op_inputs.push_back(fused_inputs[i]);
        }
        stream.set_arguments(*stage.kernel, desc, args);
        desc.workGroups.global = global;
        desc.workGroups.local = local;
        kernel_dump_info.add_entry_point(stage.kernel->get_id());
        return stream.enqueue_kernel(*stage.kernel, desc, {}, events, /*needs_completion_event=*/false);
    }

    // Collect the fused-primitive external input memories (e.g. SwiGLU up_proj output, residual tensor)
    // for the shuffle-GEMV explicit-args stage dispatch.
    static std::vector<cldnn::memory::ptr> collect_fused_inputs(const cldnn::primitive_inst& instance) {
        std::vector<cldnn::memory::ptr> fused_inputs;
        if (instance.has_fused_primitives()) {
            const size_t count = instance.get_fused_mem_count();
            fused_inputs.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                fused_inputs.push_back(instance.fused_memory(i));
            }
        }
        return fused_inputs;
    }

    // Cached direct dnnl::matmul WOQ primitive keyed by (gguf type, M, K, N). K/N are static per node
    // so in practice the cache holds one entry per distinct prefill M.
    struct GgufMatmul {
        dnnl::matmul prim;
        dnnl::matmul::primitive_desc pd;
        dnnl::memory::desc src_md;
        dnnl::memory::desc wei_md;
        dnnl::memory::desc dst_md;
        dnnl::memory::desc scale_md;      // weight scale [K/group, N]  — f16
        dnnl::memory::desc act_scale_md;  // activation scale [M, 1]    — f16, W4A8 mode only
        dnnl::memory::desc zp_md;         // weight zero-point [K/group, N] — u8, asymmetric mode
        // Per-binary-post-op src1 descriptor + its FC dependency index (parallel to the binary ops, in
        // post-op order). Empty when no fused ops. Used to bind DNNL_ARG_ATTR_MULTIPLE_POST_OP src1.
        std::vector<std::pair<int, dnnl::memory::desc>> binary_post_op_srcs;
    };
    struct MmKey {
        int et;
        int m;
        int k;
        int n;
        uint64_t fused_sig;
        int f16;
        int dq;    // 1 = W4A8 DQ-activation mode, 0 = WOQ f16-activation mode
        int asym;  // 1 = asymmetric u4+ZP, 0 = symmetric s8 (no ZP)
        int rg;    // requant_group: 32 (non-shuffle small-block) or 128 (K-block / shuffled)
        bool operator==(const MmKey& o) const {
            return et == o.et && m == o.m && k == o.k && n == o.n &&
                   fused_sig == o.fused_sig && f16 == o.f16 && dq == o.dq && asym == o.asym && rg == o.rg;
        }
    };
    struct MmKeyHash {
        size_t operator()(const MmKey& k) const {
            size_t h = std::hash<int>()(k.et);
            h = h * 31 + std::hash<int>()(k.m);
            h = h * 31 + std::hash<int>()(k.k);
            h = h * 31 + std::hash<int>()(k.n);
            h = h * 31 + std::hash<uint64_t>()(k.fused_sig);
            h = h * 31 + std::hash<int>()(k.f16);
            h = h * 31 + std::hash<int>()(k.dq);
            h = h * 31 + std::hash<int>()(k.asym);
            h = h * 31 + std::hash<int>()(k.rg);
            return h;
        }
    };
    mutable cldnn::LruCache<MmKey, std::shared_ptr<GgufMatmul>, MmKeyHash> m_matmul_cache{64};

    GgufMatmul& get_matmul(element::Type_t et,
                           int M,
                           int K,
                           int N,
                           dnnl::memory::data_type src_dt,
                           dnnl::memory::data_type dst_dt,
                           dnnl::engine& eng,
                           const std::vector<GgufFusedOp>& fused_ops,
                           bool use_f16,
                           bool use_dq_act,
                           bool use_asym,
                           int requant_group) {
        MmKey key{static_cast<int>(et), M, K, N, gguf_fused_signature(fused_ops),
                  use_f16 ? 1 : 0, use_dq_act ? 1 : 0, use_asym ? 1 : 0, requant_group};
        if (m_matmul_cache.has(key)) {
            return *m_matmul_cache.get(key);
        }

        auto k = std::make_shared<GgufMatmul>();
        k->dst_md = dnnl::memory::desc({M, N}, dst_dt, dnnl::memory::format_tag::ab);

        dnnl::primitive_attr attr;
        const int grouped = (1 << 0) | (1 << 1);  // grouped scale mask for 2-D tensors

        if (use_f16) {
            // f16 prefill (OV_GPU_GGUF_PREFILL_F16=1 or K-block shuffle): the transcode kernel fully
            // dequantised the GGUF weight into an f16 scratchpad, so this is a plain f16 x f16 GEMM with
            // NO WOQ scales. Fixed weight layout [K, N] as `ba` -> physical [N, K] (matches the transcode
            // kernel's f16 write).
            k->src_md = dnnl::memory::desc({M, K}, src_dt, dnnl::memory::format_tag::ab);
            k->wei_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ba);
        } else if (use_dq_act) {
            // W4A8 DP4A path: INT8 activation × u4/u8/s8 weight.
            // - Asymmetric 4-bit (u4, Q4_K/Q4_0/Q4_1): jit:gemm:any s8×u4 with u8 ZP.
            // - Symmetric 8-bit (s8, Q5_K/Q6_K/IQ2/Q8_0 FC): jit:gemm:any s8×s8 (dy_quant_enabled).
            const auto tgt_l = transcode_target(et, static_cast<size_t>(N));
            const auto w_dt = !use_asym        ? dnnl::memory::data_type::s8   // s8 symmetric
                             : tgt_l.to_i4     ? dnnl::memory::data_type::u4   // u4 asymmetric 4-bit
                             : dnnl::memory::data_type::u8;                    // u8 asymmetric 8-bit
            k->src_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab);
            k->wei_md = dnnl::memory::desc({K, N}, w_dt, dnnl::memory::format_tag::ba);
            // Weight scale: per-REQUANT_GROUP x per-N f16.
            k->scale_md = dnnl::memory::desc({K / requant_group, N},
                                             dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
            // Activation scale: per-token [M, 1] f16.
            k->act_scale_md = dnnl::memory::desc({M, 1}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
            attr.set_scales(DNNL_ARG_SRC,     grouped, dnnl::memory::dims{1, K},                 dnnl::memory::data_type::f16);
            attr.set_scales(DNNL_ARG_WEIGHTS, grouped, dnnl::memory::dims{requant_group, 1}, dnnl::memory::data_type::f16);
            if (use_asym) {
                // Asymmetric u4: per-group u8 zero-point (same grouping as scale).
                k->zp_md = dnnl::memory::desc({K / requant_group, N},
                                              dnnl::memory::data_type::u8, dnnl::memory::format_tag::ab);
                attr.set_zero_points(DNNL_ARG_WEIGHTS, grouped, dnnl::memory::dims{requant_group, 1}, dnnl::memory::data_type::u8);
            }
        } else {
            // WOQ f16-activation path: f16 activation × u4/u8/s8 weight + fpmath_mode f16.
            // - Asymmetric 4-bit (u4): f16×u4 with weight scale + u8 ZP.
            // - Asymmetric 8-bit (u8, LM Head Q8_0): f16×u8 with weight scale + u8 ZP; f16 precision.
            // - Symmetric 8-bit (s8): f16×s8 with weight scale, no ZP.
            const auto tgt_l = transcode_target(et, static_cast<size_t>(N));
            const auto w_dt = !use_asym        ? dnnl::memory::data_type::s8   // s8 symmetric
                             : tgt_l.to_i4     ? dnnl::memory::data_type::u4   // u4 asymmetric 4-bit
                             : dnnl::memory::data_type::u8;                    // u8 asymmetric 8-bit (LM Head)
            k->src_md = dnnl::memory::desc({M, K}, src_dt, dnnl::memory::format_tag::ab);
            k->wei_md = dnnl::memory::desc({K, N}, w_dt, dnnl::memory::format_tag::ba);
            k->scale_md = dnnl::memory::desc({K / requant_group, N},
                                             dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
            attr.set_scales(DNNL_ARG_WEIGHTS, grouped, dnnl::memory::dims{requant_group, 1}, dnnl::memory::data_type::f16);
            if (use_asym) {
                k->zp_md = dnnl::memory::desc({K / requant_group, N},
                                              dnnl::memory::data_type::u8, dnnl::memory::format_tag::ab);
                attr.set_zero_points(DNNL_ARG_WEIGHTS, grouped, dnnl::memory::dims{requant_group, 1}, dnnl::memory::data_type::u8);
            }
        }

        // Express the fused SwiGLU/residual ops as matmul post-ops so the prefill path produces the
        // same result as the decode FUSED_OPS kernels (and avoids the unfused-subgraph fallback). The
        // binary post-op src1 is the EXTERNAL peer tensor [M, N], same dtype/layout as dst.
        if (!fused_ops.empty()) {
            dnnl::post_ops po;
            for (const auto& op : fused_ops) {
                if (op.is_binary) {
                    dnnl::memory::desc src1_md({M, N}, dst_dt, dnnl::memory::format_tag::ab);
                    po.append_binary(op.alg, src1_md);
                    k->binary_post_op_srcs.emplace_back(op.ext_input_dep, src1_md);
                } else {
                    po.append_eltwise(op.alg, op.alpha, op.beta);
                }
            }
            attr.set_post_ops(po);
        }

        k->pd = dnnl::matmul::primitive_desc(eng, k->src_md, k->wei_md, k->dst_md, attr);
        // Bind the scratchpad through the pd's actual descriptors (dnnl may choose an internal weight
        // layout that differs from the requested ba tag); using the requested md can mis-read the bytes.
        k->wei_md = k->pd.weights_desc();
        k->src_md = k->pd.src_desc();
        k->dst_md = k->pd.dst_desc();
        k->prim = dnnl::matmul(k->pd);
        m_matmul_cache.add(key, k);
        return *m_matmul_cache.get(key);
    }

    cldnn::event::ptr execute_transcode_plus_onednn_woq(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance, size_t M) {
        const auto& params = *instance.get_impl_params();
        const auto& in1 = params.get_input_layout(1);
        const auto& shape_w = in1.get_shape();
        const int N = static_cast<int>(shape_w[0]);
        const int K = static_cast<int>(shape_w[1]);
        const auto et = static_cast<element::Type_t>(in1.data_type);
        // Small-block (Q4_0/Q4_1/Q8_0) shuffle nodes transcode one 256-elem SG super-block per work-item
        // (block_elem 256), matching the shuffle-aware decode in fc_gguf_transcode.cl; every other node
        // uses its native block granularity.
        const bool small_shuffle = (m_q4_0_shuffle || m_q4_1_shuffle || m_q8_0_shuffle);
        const int block_elem = small_shuffle ? 256 : static_cast<int>(element::Type(et).block_elem_count());
        const int blocks_per_row = K / block_elem;

        auto& stream = instance.get_network().get_stream();
        auto& engine = instance.get_network().get_engine();
        auto* fc_inst = dynamic_cast<cldnn::fully_connected_inst*>(&instance);

        // Draw the transcode weight + scale from the shared, per-stream scratchpad (grown to the largest
        // FC seen and reused in-order across all GGUF FC nodes) instead of a persistent per-node second
        // weight copy. Safe because oneDNN forces an in-order queue (get_onednn_stream asserts it): this
        // node's matmul finishes reading the scratch before the next node's transcode overwrites it.
        if (!m_transcode_arena) {
            m_transcode_arena = get_transcode_arena(&engine);
        }
        const auto tgt = transcode_target(et, static_cast<size_t>(N));
        // K-block shuffles (Q4_K/Q5_K/Q6_K) no longer force the f16 path: they now go through the
        // requant path (u4 asymmetric for Q4_K, s8 symmetric for Q5_K/Q6_K), using the shuffle-aware
        // transcode decoder (GGUF_SHUFFLE=1) for coalesced reads. Only the explicit flag forces f16.
        const bool use_f16 = m_prefill_f16;
        // requant_group=128 when block_elem=256: matches NNCF group=128, 4x fewer scale/ZP ops in GEMM.
        const int requant_group = gguf_requant_group(block_elem);
        const auto w_dt = use_f16 ? data_types::f16
                         : !tgt.asymmetric ? data_types::i8   // s8 symmetric (Q5_K/Q6_K/Q8_0/IQ2*)
                         : tgt.to_i4       ? data_types::u4   // u4 asymmetric 4-bit (Q4_K/Q4_0/Q4_1/...)
                         : data_types::u8;                    // u8 asymmetric 8-bit (not used currently)
        const size_t num_groups = static_cast<size_t>(K) / requant_group;
        const cldnn::layout w_layout(ov::Shape{static_cast<size_t>(N), static_cast<size_t>(K)}, w_dt, cldnn::format::bfyx);
        const cldnn::layout s_layout(ov::Shape{num_groups, static_cast<size_t>(N)}, data_types::f16, cldnn::format::bfyx);
        const cldnn::layout zp_layout(ov::Shape{num_groups, static_cast<size_t>(N)}, data_types::u8, cldnn::format::bfyx);

        cldnn::memory::ptr w_scratch;   // packed u4/u8 weight [N, K]  (asymmetric unsigned)
        cldnn::memory::ptr s_scratch;   // f16 scale [K/group, N]
        cldnn::memory::ptr zp_scratch;  // u8  zero-point [K/group, N]  (NEW: asymmetric)
        {
            const auto alloc_type = engine.get_preferred_memory_allocation_type();
            std::lock_guard<std::mutex> lk(m_transcode_arena->mtx);
            auto& slot = m_transcode_arena->per_stream[&stream];
            // Grow-only: (re)allocate when absent or too small. Growth frees the old buffer, which a
            // prior node's matmul may still be reading on the device, so finish the queue first. After
            // the first prefill pass the scratch is at its high-water mark and never grows again.
            if (!slot.weight || slot.weight->size() < w_layout.bytes_count()) {
                if (slot.weight) {
                    stream.finish();
                }
                slot.weight = engine.allocate_memory(w_layout, alloc_type, /*reset=*/false);
            }
            if (!slot.scale || slot.scale->size() < s_layout.bytes_count()) {
                if (slot.scale) {
                    stream.finish();
                }
                slot.scale = engine.allocate_memory(s_layout, alloc_type, /*reset=*/false);
            }
            // Grow-only zero-point scratch (same num_groups x N shape, u8 = half the bytes of f16 scale).
            if (!slot.zp || slot.zp->size() < zp_layout.bytes_count()) {
                if (slot.zp) {
                    stream.finish();
                }
                slot.zp = engine.allocate_memory(zp_layout, alloc_type, /*reset=*/false);
            }
            // View the (possibly larger) high-water buffers as this node's exact layout.
            w_scratch  = engine.reinterpret_buffer(*slot.weight, w_layout);
            s_scratch  = engine.reinterpret_buffer(*slot.scale,  s_layout);
            zp_scratch = engine.reinterpret_buffer(*slot.zp,     zp_layout);
        }

        // The FC weight memory is the (possibly shuffled) weight the transform produced; the transcode
        // kernel reads it either as native blocks or as the SG-shuffle layout (GGUF_SHUFFLE JIT).
        auto weight_mem = fc_inst ? fc_inst->weights_memory() : instance.dep_memory_ptr(1);

        // Stage 1: transcode GGUF blocks -> {f16 weight (shuffle / f16 prefill) OR packed u4/u8 weight
        // + f16 per-group scale + u8 per-group zero-point}.
        // One work-item per (n, block): global = [ceil(N/SG)*SG, blocks/row, 1].
        const size_t n_global = ((static_cast<size_t>(N) + GGUF_GEMV_SG_SIZE - 1) / GGUF_GEMV_SG_SIZE) * GGUF_GEMV_SG_SIZE;
        cldnn::event::ptr transcode_ev = execute_stage(events,
                                                       instance,
                                                       *transcode_stage,
                                                       /*inputs=*/{weight_mem},
                                                       /*outputs=*/{w_scratch, s_scratch, zp_scratch},
                                                       /*global=*/{n_global, static_cast<size_t>(blocks_per_row), 1},
                                                       /*local=*/{GGUF_GEMV_SG_SIZE, 1, 1});

        // Stage 2: direct dnnl::matmul WOQ consuming the scratchpad. The OneDNN stream shares the same
        // in-order OCL queue, so submission order serialises it after the transcode kernel; pass the
        // transcode event as the dependency for the (later) returned event ordering.
        auto& dnn_stream = stream.get_onednn_stream();
        auto& dnn_engine = instance.get_network().get_engine().get_onednn_engine();
        const auto src_dt = onednn::convert_data_type(params.get_input_layout(0).data_type);
        const auto dst_dt = onednn::convert_data_type(params.get_output_layout(0).data_type);

        // SwiGLU/residual fused ops -> matmul post-ops, so prefill matches the decode FUSED_OPS path.
        std::vector<GgufFusedOp> fused_ops;
        bool fused_supported = true;
        const bool has_fused = extract_gguf_fused_ops(params, fused_ops, fused_supported);
        OPENVINO_ASSERT(!instance.has_fused_primitives() || (has_fused && fused_supported),
                        "[GPU] FCGGUFOpt: unsupported fused op for the prefill matmul post-op path.");
        // DQ activation: only when W4A8 path is active (not f16, and DQ enabled).
        // LM Head heuristic: large vocabulary output (N ≥ 50000, e.g. Qwen3=151936, Llama2=32000).
        // LM Head must preserve f16 activation precision for logit accuracy; use WOQ (no INT8 DQ).
        const bool is_lm_head = (static_cast<size_t>(N) >= 50000);
        const bool run_dq = !use_f16 && m_use_dq_act && !is_lm_head;

        // Optionally grow and fill activation scratch (INT8 [M, K] + f16 scale [M, 1]).
        cldnn::memory::ptr act_q_scratch;   // INT8 activation
        cldnn::memory::ptr act_s_scratch;   // f16 per-token scale
        if (run_dq) {
            const cldnn::layout aq_layout(ov::Shape{static_cast<size_t>(M), static_cast<size_t>(K)},
                                          data_types::i8, cldnn::format::bfyx);
            const cldnn::layout as_layout(ov::Shape{static_cast<size_t>(M), 1},
                                          data_types::f16, cldnn::format::bfyx);
            const auto alloc_type = engine.get_preferred_memory_allocation_type();
            {
                std::lock_guard<std::mutex> lk(m_transcode_arena->mtx);
                auto& slot = m_transcode_arena->per_stream[&stream];
                if (!slot.act_int8 || slot.act_int8->size() < aq_layout.bytes_count()) {
                    if (slot.act_int8) stream.finish();
                    slot.act_int8  = engine.allocate_memory(aq_layout, alloc_type, /*reset=*/false);
                }
                if (!slot.act_scale || slot.act_scale->size() < as_layout.bytes_count()) {
                    if (slot.act_scale) stream.finish();
                    slot.act_scale = engine.allocate_memory(as_layout, alloc_type, /*reset=*/false);
                }
                act_q_scratch = engine.reinterpret_buffer(*slot.act_int8,  aq_layout);
                act_s_scratch = engine.reinterpret_buffer(*slot.act_scale, as_layout);
            }
            // Activation DQ stage: FP16 [M, K] -> INT8 [M, K] + f16 scale [M, 1].
            const size_t dq_global = static_cast<size_t>(M) * GGUF_ACT_DQ_WG_SIZE;
            execute_stage({transcode_ev},
                          instance,
                          *act_dq_stage,
                          /*inputs=*/ {instance.dep_memory_ptr(0)},
                          /*outputs=*/{act_q_scratch, act_s_scratch},
                          /*global=*/ {dq_global, 1, 1},
                          /*local=*/  {static_cast<size_t>(GGUF_ACT_DQ_WG_SIZE), 1, 1});
        }

        auto& mm = get_matmul(et, static_cast<int>(M), K, N, src_dt, dst_dt, dnn_engine, fused_ops,
                              use_f16, run_dq, tgt.asymmetric, requant_group);

        // Select the activation source: INT8 scratch (W4A8) or original FP16 buffer (WOQ/f16).
        auto src_mem_ptr = run_dq ? act_q_scratch : instance.dep_memory_ptr(0);
        auto src = src_mem_ptr->get_onednn_memory(mm.src_md);
        auto dst = instance.output_memory_ptr(0)->get_onednn_memory(mm.dst_md);
        auto wei = w_scratch->get_onednn_memory(mm.wei_md);

        std::unordered_map<int, dnnl::memory> args{
            {DNNL_ARG_SRC, src},
            {DNNL_ARG_WEIGHTS, wei},
            {DNNL_ARG_DST, dst},
        };
        if (run_dq) {
            // W4A8: bind activation scale + weight scale + optional weight ZP (asymmetric only).
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0,       act_s_scratch->get_onednn_memory(mm.act_scale_md)});
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,     s_scratch->get_onednn_memory(mm.scale_md)});
            if (tgt.asymmetric) {
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_scratch->get_onednn_memory(mm.zp_md)});
            }
        } else if (!use_f16) {
            // WOQ f16-activation: bind weight scale + optional weight ZP (asymmetric only).
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,     s_scratch->get_onednn_memory(mm.scale_md)});
            if (tgt.asymmetric) {
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_scratch->get_onednn_memory(mm.zp_md)});
            }
        }
        // Bind each binary post-op's EXTERNAL src1 (up_proj output / residual). Post-op indices count
        // ALL post-ops (incl. unary eltwise), so po_idx tracks the running position; the binary peer
        // memories are the node's fused memories in occurrence order (same convention as the onednn FC
        // base impl: instance.fused_memory(i)).
        if (!mm.binary_post_op_srcs.empty()) {
            size_t binary_i = 0;
            int po_idx = 0;
            for (const auto& op : fused_ops) {
                if (op.is_binary) {
                    const auto& src1_md = mm.binary_post_op_srcs[binary_i].second;
                    auto peer = instance.fused_memory(binary_i)->get_onednn_memory(src1_md);
                    args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_idx) | DNNL_ARG_SRC_1, peer});
                    ++binary_i;
                }
                ++po_idx;
            }
        }
        mm.prim.execute(dnn_stream, args);

        // The matmul (last op) returns no cldnn event; on the shared in-order queue, downstream
        // consumers observe its output. Return the transcode event for dependency tracking; if the
        // output is a network output, force completion so host reads see the matmul result.
        if (instance.needs_completion_event()) {
            stream.finish();
        }
        return transcode_ev;
    }

    // Q4_K / Q6_K weight-shuffle decode: the FC weight is already the SG-shuffled buffer (produced by
    // RepackGGUFWeightsShuffle), so the sub-group-block-read GEMV reads it with coalesced loads. Each
    // subgroup (SG_SIZE = 16 lanes) owns one row-group; each lane owns one output row. Dispatch:
    //   global = [ceil(N/16)*16, M, 1], local = [16, 1, 1].
    // Activation is f16 (INPUT0), read directly (no prequant scratch). Q4_K and Q6_K use their own
    // kernel entry point (q4k_sg_stage / q6k_sg_stage). The weight is passed as the plane-separated
    // shuffle buffer via the FC weight memory; for Q6_K the kernel internally slices it into the
    // pql/pqh/ps/pd planes (matching RepackGGUFWeightsShuffle's plane offsets).
    cldnn::event::ptr execute_shuffle_gemv(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance, size_t M) {
        const auto& params = *instance.get_impl_params();
        const auto& in1 = params.get_input_layout(1);
        const size_t N = in1.get_shape()[0];
        const size_t K = in1.get_shape()[1];
        const size_t row_groups = (N + GGUF_GEMV_SG_SIZE - 1) / GGUF_GEMV_SG_SIZE;
        // K-split (occupancy) factor: KSPLIT sub-groups per work-group, each on a strided K-subset,
        // reduced in SLM. MUST match the KSPLIT JIT constant (gguf_shuffle_choose_ksplit) so the grid
        // and the kernel agree. Placed on the 3rd dispatch dim so the BM (M) dim is untouched; the
        // work-group is OPG x 1 x KSPLIT -> KSPLIT sub-groups per row-group. KSPLIT=1 reproduces the
        // original [ceil(N/16)*16, M, 1] / [16, 1, 1] geometry exactly.
        const size_t ksplit = static_cast<size_t>(gguf_shuffle_choose_ksplit(N, K));

        auto* fc_inst = dynamic_cast<cldnn::fully_connected_inst*>(&instance);
        auto act = instance.dep_memory_ptr(0);
        auto weight = fc_inst ? fc_inst->weights_memory() : instance.dep_memory_ptr(1);
        auto out = instance.output_memory_ptr(0);

        Stage& sg_stage = m_q4k_shuffle   ? *q4k_sg_stage
                        : m_q5k_shuffle   ? *q5k_sg_stage
                        : m_q6k_shuffle   ? *q6k_sg_stage
                        : m_q4_0_shuffle  ? *q4_0_sg_stage
                        : m_q4_1_shuffle  ? *q4_1_sg_stage
                                          : *q8_0_sg_stage;
        const bool needs_shape_info = is_dynamic();
        return execute_stage(events,
                             instance,
                             sg_stage,
                             /*inputs=*/{act, weight},
                             /*outputs=*/{out},
                             /*global=*/{row_groups * GGUF_GEMV_SG_SIZE, M, ksplit},
                             /*local=*/{GGUF_GEMV_SG_SIZE, 1, ksplit},
                             /*fused_inputs=*/collect_fused_inputs(instance),
                             /*needs_shape_info=*/needs_shape_info);
    }

#endif  // ENABLE_ONEDNN_FOR_GPU
};

}  // namespace

std::unique_ptr<primitive_impl> FCGGUFOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<fully_connected>());
    return std::make_unique<FCGGUFOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::FCGGUFOptImpl)
