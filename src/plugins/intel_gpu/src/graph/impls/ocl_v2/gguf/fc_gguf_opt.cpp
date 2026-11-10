// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_gguf_opt.hpp"
#include "fc_gguf_shuffle_gen.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>

#include "../primitive_ocl_base.hpp"
#include "../utils/fused_ops_jitter.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
#include "fully_connected_inst.h"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/util/log.hpp"

#ifdef ENABLE_CM_FOR_GPU
#    include "../../cm/utils/kernels_db.hpp"  // cm::SourcesDB::get_kernel_template() for .cm files
#endif

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

// Which kernel runs the prefill (M > threshold) of one FC node; decided once per node in the impl constructor.
enum class PrefillKernel : uint8_t {
    ocl = 0,     // OCL dpas GEMM (fc_gguf_gemm_opt.cl)
    cm = 1,      // CM dpas GEMM (gemm_q*_full.cm), comparison aid only
    onednn = 2,  // transcode + oneDNN matmul (fallback)
    none = 3,    // GEMV only
};

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

// GGUF_GEMV_SG_SIZE, gguf_shuffle_choose_ksplit, FCGGUFShuffleGenerator and its per-format
// subclasses (FCGGUFQ{4k,5k,6k,40,41,80}SgGenerator) are in the shared header:
//   fc_gguf_shuffle_gen.hpp

#ifdef ENABLE_ONEDNN_FOR_GPU
// Q4_K / Q6_K "weight shuffle" path: when enabled, the weight was reordered once (compile_model) into
// the SG-transposed plane-separated layout that the sub-group-block-read GEMV kernels (fc_gguf_gemv_opt.cl) and the shuffle-aware prefill transcode consume with fully coalesced weight
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

// Opt-in f16 prefill (OV_GPU_GGUF_PREFILL_F16=1, default on). When on, the prefill transcode kernel
// fully dequantises the GGUF weight into an f16 scratchpad and the direct dnnl::matmul runs as a plain
// f16 x f16 GEMM (no WOQ scales) instead of the default i4/i8 weight-only-quantised path. This trades
// ~2-4x more weight bandwidth for reference-precision compute; useful for numeric comparison against
// the i8/i4 WOQ path. The transcode generator reads this directly (like gguf_shuffle_enabled) and the
// impl mirrors it into m_prefill_f16 so both sides agree.
bool gguf_prefill_f16_enabled() {
    if (const char* env = std::getenv("OV_GPU_GGUF_PREFILL_F16")) {
        return std::atol(env) != 0;
    }
    return true;  // ON by default
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

size_t gguf_act_dq_group_size(size_t K) {
    if (const char* env = std::getenv("OV_GPU_GGUF_PREFILL_DQ_GROUP_SIZE")) {
        const long v = std::atol(env);
        if (v >= 32 && static_cast<size_t>(v) <= K && K % static_cast<size_t>(v) == 0)
            return static_cast<size_t>(v);
    }
    return K;  // DynamicQuantize's UINT64_MAX/per-token layout: one group per row.
}

bool gguf_prefill_dq_act_asymmetric() {
    if (const char* env = std::getenv("OV_GPU_GGUF_PREFILL_DQ_ASYMMETRIC"))
        return std::atol(env) != 0;
    return false;
}

void validate_gguf_act_dq_config(size_t K, size_t group, bool asymmetric) {
    OPENVINO_ASSERT(group >= 32 && group <= K && K % group == 0,
                    "[GPU] GGUF activation DQ requires group size >= 32 dividing K.");
    // Asymmetric activation DQ is now supported per-group (group=32/64/128) as well as
    // per-token (group=K): the fc_gguf_act_dq kernel emits one f16 scale + u8 ZP per
    // [token, group] and the matmul binds them as grouped DNNL_ARG_SRC scales/zero-points.
    (void)asymmetric;
}

// Returns the effective activation DQ group size for the W4A8 prefill path.
//
// Problem: when activation DQ is grouped (group < K) AND the weight is asymmetric (u4+ZP),
// oneDNN's fast jit gemm_t requires precomputed B-group-sums — i.e.
//
//   gs[k_g, n] = sum_{k in group_k_g} W_q[k, n]   (per K-group, per N column)
//
// — to evaluate the ZP-weight cross term in the W4A8 expansion without a second pass.
// We don't supply these sums, so gemm_t returns `unimplemented` and oneDNN silently falls
// back to the sequential ref_t kernel, which is ~100x slower and appears as a hang.
//
// Root cause (gemmstone, pd.cpp):
//   needsBGroupSums = aOffset(=Calc, weight has ZP) && quantized2DA(bScale2D=true)
//   pd.cpp: if (needsBGroupSums && !hasGroupSumsB) return unimplemented;
//
// Fix: if the requested group < K would trigger this rejection, silently downgrade to
// per-token (group = K). Per-token activation: bScale2D() is false (bqGroupK=K covers
// the full K range), quantized2DA()=false, needsBGroupSums()=false → fast path accepted.
// The per-token kernel variant is always compiled (it's the default when DQ_GROUP_SIZE
// env is unset), so no extra compilation cost.
size_t gguf_act_dq_safe_group_size(size_t K, element::Type_t et, size_t N) {
    const size_t group = gguf_act_dq_group_size(K);
    if (group < K && transcode_target(et, N).asymmetric) {
        // Would trigger needsBGroupSums → auto-downgrade to per-token (group=K).
        GPU_DEBUG_LOG << "[FCGGUFOpt] act_dq group=" << group
                      << " + asymmetric weight would trigger oneDNN needsBGroupSums "
                         "(ref_t fallback). Auto-downgrading to per-token (group=" << K << ").\n";
        return K;
    }
    return group;
}

// -------------------------------------------------------------------------------------------------
// Fused post-op support for the prefill (transcode + dnnl::matmul) path.
//
// Each fused op maps 1:1 to a matmul post-op:
//   activation(func, a, b)  -> append_eltwise(convert_activation_func(func), a, b)
//   eltwise(prod|sum, ext)  -> append_binary(binary_mul|binary_add, ext_md[M,N])
// EXTERNAL binary inputs are bound at execute via DNNL_ARG_ATTR_MULTIPLE_POST_OP(i)|DNNL_ARG_SRC_1.
struct GgufFusedOp {
    bool is_binary;  // false: unary eltwise(activation); true: binary(eltwise)
    dnnl::algorithm alg;
    float alpha = 0.0f;
    float beta  = 0.0f;
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
    cldnn::memory::ptr act_zp;    // u8 activation zero point [M_max, groups]
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
            make_jit_constant("FC_INPUT_RANK", in0.get_partial_shape().size()),
            make_jit_constant("FC_OUTPUT_RANK", params.output_layouts[0].get_partial_shape().size()),
            make_jit_constant("GGUF_BLOCK_ELEM", static_cast<int>(wt.block_elem_count())),
            make_jit_constant("GGUF_BLOCK_BYTES", static_cast<int>(wt.block_byte_size())),
            make_jit_constant("SG_SIZE", GGUF_GEMV_SG_SIZE),
            make_jit_constant("NROW", gguf_opt_nrow(N, K)),
            make_jit_constant(gguf_type_jit_flag(in1.data_type), 1),
        });

        // SwiGLU / residual eltwise + activation fused onto this FC apply at output-write time, so the
        // dequantized GEMV result `dequantized` is post-processed in registers (no separate kernels,
        // no unfused subgraph, output stays in the reusable memory pool).
        if (params.has_fused_primitives()) {
            const auto& out_l = params.output_layouts[0];
            const std::vector<std::string> idx_order = out_l.get_partial_shape().size() == 2
                ? std::vector<std::string>{"bm", "n", "0", "0"}
                : std::vector<std::string>{"out_b", "out_f", "n", "0"};
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

// NOTE: FCGGUFShuffleGenerator and FCGGUFQ{4k,5k,6k,40,41,80}SgGenerator are in
//       fc_gguf_shuffle_gen.hpp and shared with the CM impl.

// =================================================================================================
// Per-token / per-group INT8|UINT8 activation DQ generator (fc_gguf_act_dq.cl).
// Quantises FP16 activation [M, K] -> INT8/UINT8 [M, K] + f16 scale [M, groups] (+ u8 ZP [M, groups]
// when asymmetric). Symmetric maps by max(|v|); asymmetric maps [min, max] -> [0, 255] per group,
// matching DynamicQuantize (dynamic_quantize_gpu_opt). group == K reproduces the per-token contract.
// The token dimension is dispatched as GWS.y, the group as GWS.x / ACT_DQ_WG_SIZE.
// =================================================================================================
constexpr int GGUF_ACT_DQ_WG_SIZE = 256;

class FCGGUFActDQGenerator : public KernelGenerator {
public:
    FCGGUFActDQGenerator() : KernelGenerator("fc_gguf_act_dq") {}

protected:
    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override {
        const size_t K = params.input_layouts[1].get_shape()[1];
        const size_t N = params.input_layouts[1].get_shape()[0];
        const auto et = static_cast<element::Type_t>(params.input_layouts[1].data_type);
        // Use the safe group size: downgraded to K when asymmetric weight would needsBGroupSums.
        const size_t group = gguf_act_dq_safe_group_size(K, et, N);
        // group size + sym/asym change the JIT body, so they must be part of the cache key.
        return get_kernel_name() + "_K" + std::to_string(K) + "_G" + std::to_string(group) +
               (gguf_prefill_dq_act_asymmetric() ? "_asym" : "_sym");
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);
        const size_t K = params.input_layouts[1].get_shape()[1];
        const size_t N = params.input_layouts[1].get_shape()[0];
        const auto et = static_cast<element::Type_t>(params.input_layouts[1].data_type);
        // Use the safe group size: downgraded to K when asymmetric weight would needsBGroupSums.
        const size_t group = gguf_act_dq_safe_group_size(K, et, N);
        validate_gguf_act_dq_config(K, group, gguf_prefill_dq_act_asymmetric());
        jit.add({
            make_jit_constant("K_SIZE",         static_cast<int>(K)),
            make_jit_constant("ACT_DQ_WG_SIZE", GGUF_ACT_DQ_WG_SIZE),
            make_jit_constant("ACT_DQ_OUTPUT_VAL_MAX", 127.0f),
            make_jit_constant("ACT_DQ_MIN_VAL", 0.003f),
            make_jit_constant("ACT_DQ_GROUP_SIZE", static_cast<int>(group)),
            make_jit_constant("ACT_DQ_NUM_GROUPS", static_cast<int>(K / group)),
            make_jit_constant("ACT_DQ_ASYMMETRIC", gguf_prefill_dq_act_asymmetric() ? 1 : 0),
        });
        return jit;
    }

    // Args are supplied explicitly by execute_transcode_plus_onednn_woq.
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams&) const override { return {}; }
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {}};
    }
};

// =================================================================================================
// Prefill GEMM (OCL dpas kernels fc_gguf_gemm_opt.cl, or the CM kernels gemm_q*_full.cm for comparison).
//
// Both consume the SG-shuffled weight layout produced by RepackGGUFWeightsShuffle (the same buffer the
// decode shuffle GEMV reads), keep fp32 accumulators and produce the fp16/fp32 output directly.
// =================================================================================================
constexpr int GGUF_CM_TOKENS_PER_TILE = 8;  // dpas RepeatCount (fixed by the ISA)

// f32 -> f16 activation conversion pre-pass for the CM path (convert_act_f32_to_f16.cm). CONV_ACT_BLOCK
// divides 256 (the K-tiling granularity), so M*K is always an exact multiple of it.
constexpr int CONV_ACT_BLOCK = 64;  // elements converted per thread
constexpr int CONV_ACT_LOCAL = 16;  // threads per work-group

#ifdef ENABLE_CM_FOR_GPU
struct CmGemmConfig {
    int row_groups;    // N-dimension blocking (ROW_GROUPS)
    int token_groups;  // token dimension register blocking (TOKEN_GROUPS)
    int token_local;   // threads per work-group (TOKEN_LOCAL) — CORRECTNESS requirement
    int row_blocks;    // ROW_BLOCKS (additional N-block split within the work-group)
    int slm_jblk;      // sub-blocks published per COOP_SLM phase (or -1 = kernel default)
    int pf_a;          // PF_A prefetch flag (or -1 = kernel default)
    int a_outer;       // A_OUTER loop order flag
};

CmGemmConfig pick_cm_gemm_config(const RuntimeParams& params, size_t K) {
    const auto arch = params.get_device_info().arch;
    const bool is_xe3 = (arch >= gpu_arch::xe3);
    (void)K;  // no PTL long-K promotion any more, see the comment below

    if (is_xe3) {
        // PTL (Arc B390 iGPU, Xe3): ADOPTED config from the standalone
        // cm_gguf_gemm_kernel/ ab_lowbit_config.py PTL round-2 tuning.
        // ROW_GROUPS=4/ROW_BLOCKS=1/TOKEN_LOCAL=16 measured 1.01-1.63x
        // faster than this older RG=2/RB=2/TL=32 tile, across ALL SIX
        // quants and every shape (K 1024..16384, token_len 64..8192,
        // interleaved min-of-N -- see cm_kernel_dev.md repo memory / the
        // "PTL round 2" notes). This branch had gone stale: it still
        // shipped the PRE-round-2 tile, which is why re-measuring
        // gemm_q41_full on the real PTL box (ab-style interleaved A/B)
        // showed the old branch 1.07-1.43x SLOWER than the adopted one at
        // K/N=4096/4096..12288/4096, token_len=1024 -- a config mismatch,
        // NOT register spill (both variants build to <=121/128 registers,
        // no spill, with or without post-ops; verified via
        // -mCM_printregusage on the actual Arc B390).
        // Do NOT re-add the old RG=2 tile's "K>=10240 -> TOKEN_GROUPS=4"
        // long-K promotion on top of this: RG*TG would become 8 KB of
        // accumulators and spill (measured 0.19-0.27x). RG=4 already wins
        // at long K directly (1.54-1.63x at K=12288/16384).
        return CmGemmConfig{/*row_groups=*/4,
                            /*token_groups=*/2,
                            /*token_local=*/16,
                            /*row_blocks=*/1,
                            /*slm_jblk=*/4,
                            /*pf_a=*/1,
                            /*a_outer=*/1};
    }
    // BMG (Arc B580, Xe2) — default, tuned for 160 XVE @ 2.9 GHz.
    return CmGemmConfig{/*row_groups=*/2,
                        /*token_groups=*/4,
                        /*token_local=*/8,
                        /*row_blocks=*/1,
                        /*slm_jblk=*/-1,
                        /*pf_a=*/-1,
                        /*a_outer=*/1};
}
#endif

// Format index shared by the OCL and CM GEMM stage tables; -1 = no GEMM kernel for this format.
int gguf_gemm_fmt_index(element::Type_t t) {
    switch (t) {
    case element::Type_t::gguf_q4_0:
        return 0;
    case element::Type_t::gguf_q4_1:
        return 1;
    case element::Type_t::gguf_q4_k:
        return 2;
    case element::Type_t::gguf_q5_k:
        return 3;
    case element::Type_t::gguf_q6_k:
        return 4;
    case element::Type_t::gguf_q8_0:
        return 5;
    default:
        return -1;
    }
}

constexpr size_t GGUF_GEMM_FORMATS = 6;
constexpr const char* kOclGemmKernels[GGUF_GEMM_FORMATS] =
    {"fc_gguf_gemm_q4_0", "fc_gguf_gemm_q4_1", "fc_gguf_gemm_q4k", "fc_gguf_gemm_q5k", "fc_gguf_gemm_q6k", "fc_gguf_gemm_q8_0"};
// Format-selection define of each entry in fc_gguf_gemm_opt.cl (same order as kOclGemmKernels).
constexpr const char* kOclGemmFormatDefines[GGUF_GEMM_FORMATS] =
    {"GG_FMT_Q4_0", "GG_FMT_Q4_1", "GG_FMT_Q4_K", "GG_FMT_Q5_K", "GG_FMT_Q6_K", "GG_FMT_Q8_0"};
constexpr const char* kCmGemmKernels[GGUF_GEMM_FORMATS] =
    {"gemm_q40_full", "gemm_q41_full", "gemm_q4k_full", "gemm_q5k_full", "gemm_q6k_full", "gemm_q80_full"};

// OV_GPU_GGUF_PREFILL_GEMM=ocl|cm selects the PREFERRED prefill GEMM kernel type (default: ocl).
PrefillKernel gguf_prefill_gemm_requested() {
    const char* env = std::getenv("OV_GPU_GGUF_PREFILL_GEMM");
    if (env == nullptr || *env == '\0') {
        return PrefillKernel::ocl;
    }
    std::string v(env);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (v == "ocl") {
        return PrefillKernel::ocl;
    }
    if (v == "cm") {
        return PrefillKernel::cm;
    }
    OPENVINO_WARN("[GPU] OV_GPU_GGUF_PREFILL_GEMM='", env, "' is not one of ocl|cm; using ocl.");
    return PrefillKernel::ocl;
}

// OV_GPU_GGUF_PREFILL_GEMM_STRICT=1: no silent fallback for GEMM-family nodes (test / benchmark aid).
bool gguf_prefill_gemm_strict() {
    if (const char* env = std::getenv("OV_GPU_GGUF_PREFILL_GEMM_STRICT")) {
        return std::atol(env) != 0;
    }
    return false;
}

// OCL GEMM tile configuration (fc_gguf_gemm_body.cl JIT constants).
struct OclGemmConfig {
    int row_groups;    // ROW_GROUPS: 16-column output groups per sub-group
    int token_groups;  // TOKEN_GROUPS: 8-token DPAS tiles per sub-group
    int token_local;   // TOKEN_LOCAL: sub-groups per work-group
    int row_blocks;    // ROW_BLOCKS: N split inside the work-group
    int slm_jblk;      // SLM_JBLK: 32-K sub-blocks published per SLM phase
    bool large_grf;    // build with 256 GRF per thread
    int n_band;        // GG_N_BAND: work-group band swizzle (0 = plain dispatch order)
};

OclGemmConfig pick_ocl_gemm_config(const RuntimeParams& params) {
    if (params.get_device_info().arch >= gpu_arch::xe3) {
        // PTL (Arc B390), 256 GRF: every sub-group owns 32 tokens x 64 outputs. The 4-bit formats decode cheaply
        // enough to share each activation tile between two row blocks (256 tokens x 128 outputs per work-group);
        // Q5_K / Q6_K / Q8_0 run best with 256 tokens x 64 outputs per work-group.
        const auto& wl = params.input_layouts[1];
        const size_t N = wl.get_shape()[0];
        const size_t K = wl.get_shape()[1];
        const auto wt_id = wl.data_type;
        const bool four_bit = wt_id == data_types::gguf_q4_0 || wt_id == data_types::gguf_q4_1 || wt_id == data_types::gguf_q4_k;
        OclGemmConfig cfg = four_bit && N % 128 == 0 ? OclGemmConfig{4, 4, 16, 2, 4, true, 0} : OclGemmConfig{4, 4, 8, 1, 4, true, 0};
        // Once the weight matrix overflows the ~8-12 MB GPU L2 the per-token-block re-reads of the weights
        // become the limit; walking the token blocks of a ~12 MB band of outputs first keeps the band cache
        // resident. Beyond ~32 MB of weights the extra activation re-reads cost more than they save.
        const element::Type wt(wt_id);
        const size_t w_bytes = N * K / wt.block_elem_count() * wt.block_byte_size();
        constexpr size_t band_bytes = size_t{12} << 20;
        if (!four_bit && w_bytes > (size_t{16} << 20) && w_bytes <= (size_t{32} << 20)) {
            const size_t ngroups = N / (GGUF_GEMV_SG_SIZE * static_cast<size_t>(cfg.row_groups * cfg.row_blocks));
            cfg.n_band = static_cast<int>((ngroups * band_bytes + w_bytes - 1) / w_bytes);
        }
        return cfg;
    }
    // BMG (Xe2): same tile as the CM kernel; not tuned for the OCL kernel yet.
    return OclGemmConfig{2, 4, 8, 1, 4, false, 0};
}

size_t gguf_ocl_gemm_align(const RuntimeParams& params) {
    const OclGemmConfig cfg = pick_ocl_gemm_config(params);
    return GGUF_GEMV_SG_SIZE * static_cast<size_t>(cfg.row_groups * cfg.row_blocks);
}

// =================================================================================================
// OCL GEMM generator: one entry per GGUF format. All formats share fc_gguf_gemm_opt.cl (per-format tile codec,
// selected by a GG_FMT_* define) and the pipeline template in fc_gguf_gemm_body.cl.
// Args: [shape_info,] A, W, C, M, K, N [, fused inputs].
// =================================================================================================
class FCGGUFGemmGenerator : public KernelGenerator {
public:
    FCGGUFGemmGenerator(const char* kernel_name, const char* format_define)
        : KernelGenerator(kernel_name, "", "fc_gguf_gemm_opt"),
          m_format_define(format_define) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);
        jit.add(make_tensors_jit_constants(params));
        const auto& in0 = params.input_layouts[0];
        const auto& out0 = params.output_layouts[0];
        jit.add(make_type_jit_constants("INPUT0", in0.data_type));
        jit.add(make_type_jit_constants("OUTPUT", out0.data_type));

        const OclGemmConfig cfg = pick_ocl_gemm_config(params);
        jit.add({
            make_jit_constant("ROW_GROUPS", cfg.row_groups),
            make_jit_constant("TOKEN_GROUPS", cfg.token_groups),
            make_jit_constant("TOKEN_LOCAL", cfg.token_local),
            make_jit_constant("ROW_BLOCKS", cfg.row_blocks),
            make_jit_constant("SLM_JBLK", cfg.slm_jblk),
            make_jit_constant("PF_A", 1),
            make_jit_constant("GG_N_BAND", cfg.n_band),
            make_jit_constant("GG_A_F32", in0.data_type == data_types::f32 ? 1 : 0),
            make_jit_constant("GG_OUT_F32", out0.data_type == data_types::f32 ? 1 : 0),
            make_jit_constant(m_format_define, 1),
        });
        if (params.has_fused_primitives()) {
            const std::vector<std::string> idx_order = out0.get_partial_shape().size() == 2
                                                           ? std::vector<std::string>{"bm", "n", "0", "0"}
                                                           : std::vector<std::string>{"out_b", "out_f", "n", "0"};
            FusedOpsConfiguration conf = {"", idx_order, "dequantized", out0.data_type};
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
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // M
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // K
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // N
        add_fused_ops_arguments(args, params);
        return args;
    }

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        auto options = KernelGenerator::get_build_options(params);
        if (pick_ocl_gemm_config(params).large_grf) {
            options += " -cl-intel-256-GRF-per-thread";
        }
        return options;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto& in1 = params.get_input_layout(1);
            const size_t N = in1.get_shape()[0];
            const size_t K = in1.get_shape()[1];
            const size_t M = derive_bm(params.get_input_layout(0).get_shape());
            const OclGemmConfig cfg = pick_ocl_gemm_config(params);

            const size_t tokens_per_thread = GGUF_CM_TOKENS_PER_TILE * static_cast<size_t>(cfg.token_groups);
            const size_t token_slots = static_cast<size_t>(cfg.token_local) / static_cast<size_t>(cfg.row_blocks);
            const size_t ntiles = (M + tokens_per_thread - 1) / tokens_per_thread;
            const size_t ngroups = N / (GGUF_GEMV_SG_SIZE * static_cast<size_t>(cfg.row_groups) * static_cast<size_t>(cfg.row_blocks));
            const size_t gsize_y = ((ntiles + token_slots - 1) / token_slots) * static_cast<size_t>(cfg.token_local);

            // dim0 = sub-group lanes of every row-group block, dim1 = token tiles (TOKEN_LOCAL sub-groups per WG).
            auto& wgs = kd.params.workGroups;
            wgs.global = {ngroups * GGUF_GEMV_SG_SIZE, gsize_y, 1};
            wgs.local = {GGUF_GEMV_SG_SIZE, static_cast<size_t>(cfg.token_local), 1};

            kd.params.scalars.clear();
            auto push_u32 = [&](size_t v) {
                scalar_desc sd{};
                sd.t = scalar_desc::Types::UINT32;
                sd.v.u32 = static_cast<uint32_t>(v);
                kd.params.scalars.push_back(sd);
            };
            push_u32(M);
            push_u32(K);
            push_u32(N);
        }};
    }

private:
    std::string m_format_define;  // GG_FMT_* macro selecting the format section of fc_gguf_gemm_opt.cl
};

// -------------------------------------------------------------------------------------------------
// Per-node prefill kernel selection. Every function returns an empty string when the kernel can be
// used and a human readable blocker otherwise.
// -------------------------------------------------------------------------------------------------
std::string gguf_gemm_common_blocker(const RuntimeParams& params, bool shuffled_family, size_t align) {
    if (!shuffled_family) {
        return "weight is not a shuffled Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0 tensor (OV_GPU_GGUF_SHUFFLE=0, N%16 / K%256 misaligned, or no GEMM kernel for the format)";
    }
    const auto& dev = params.get_device_info();
    if (!dev.supports_immad) {
        return "device has no DPAS support (supports_immad=false)";
    }
    const auto& in0 = params.input_layouts[0];
    const auto& out0 = params.output_layouts[0];
    if (in0.data_type != data_types::f16 && in0.data_type != data_types::f32) {
        return "activation is neither f16 nor f32";
    }
    if (out0.data_type != data_types::f16 && out0.data_type != data_types::f32) {
        return "output is neither f16 nor f32";
    }
    const auto& wl = params.input_layouts[1];
    const size_t N = wl.get_shape()[0];
    if (N % align != 0) {
        return "N=" + std::to_string(N) + " is not a multiple of the GEMM channel tile " + std::to_string(align);
    }
    return {};
}

std::string gguf_ocl_gemm_blocker(const RuntimeParams& params, bool shuffled_family) {
    if (auto r = gguf_gemm_common_blocker(params, shuffled_family, gguf_ocl_gemm_align(params)); !r.empty()) {
        return r;
    }
    const auto& dev = params.get_device_info();
    if (dev.arch < gpu_arch::xe2) {
        return "GPU architecture is older than Xe2";
    }
    if (!dev.supports_intel_subgroups_short || !dev.supports_intel_subgroups_char || !dev.supports_intel_subgroup_2d_block_io ||
        !dev.supports_intel_subgroup_matrix_multiply_accumulate) {
        return "device lacks cl_intel_subgroups_short / _char, cl_intel_subgroup_2d_block_io or cl_intel_subgroup_matrix_multiply_accumulate";
    }
    return {};
}

#ifdef ENABLE_CM_FOR_GPU
// CM kernels only post-process relu / swish and binary sum / prod fused ops.
std::string gguf_cm_fused_blocker(const RuntimeParams& params) {
    if (!params.has_fused_primitives()) {
        return {};
    }
    for (const auto& fd : params.fused_desc) {
        if (fd.is_type<cldnn::reorder>()) {
            continue;
        }
        if (fd.is_type<cldnn::activation>()) {
            const auto act = fd.typed_desc<cldnn::activation>();
            if (fd.has_outer_dep() ||
                (act->activation_function != cldnn::activation_func::relu && act->activation_function != cldnn::activation_func::swish)) {
                return "fused activation is not supported by the CM kernel";
            }
            continue;
        }
        if (fd.is_type<cldnn::eltwise>()) {
            const auto elt = fd.typed_desc<cldnn::eltwise>();
            if (!fd.has_outer_dep() || (elt->mode != cldnn::eltwise_mode::sum && elt->mode != cldnn::eltwise_mode::prod) || elt->input.size() != 2 ||
                !elt->coefficients.empty() || !elt->stride.empty()) {
                return "fused eltwise is not supported by the CM kernel";
            }
            const auto& operand = params.get_input_layout(fd.outer_dep_start_idx);
            const auto& out = params.get_output_layout(0);
            if (operand.format != format::bfyx || out.format != format::bfyx || operand.data_padding || out.data_padding ||
                (operand.data_type != data_types::f16 && operand.data_type != data_types::f32)) {
                return "fused eltwise operand layout is not supported by the CM kernel";
            }
            continue;
        }
        return "fused op type is not supported by the CM kernel";
    }
    return {};
}
#endif

std::string gguf_cm_gemm_blocker([[maybe_unused]] const program_node& node, const RuntimeParams& params, bool shuffled_family) {
#ifdef ENABLE_CM_FOR_GPU
    const CmGemmConfig cm_cfg = pick_cm_gemm_config(params, params.input_layouts[1].get_shape()[1]);
    const size_t cm_align = GGUF_GEMV_SG_SIZE * static_cast<size_t>(cm_cfg.row_groups) * static_cast<size_t>(cm_cfg.row_blocks);
    if (auto r = gguf_gemm_common_blocker(params, shuffled_family, cm_align); !r.empty()) {
        return r;
    }
    const auto& prog = node.get_program();
    if (!prog.get_config().get_use_cm()) {
        return "CM is disabled (set OV_GPU_USE_CM=ON / GPU_USE_CM)";
    }
    if (!cldnn::check_cm_jit_support(prog.get_engine(), prog.get_config())) {
        return "the CM JIT is not usable with the installed driver";
    }
    return gguf_cm_fused_blocker(params);
#else
    (void)params;
    (void)shuffled_family;
    return "this build has no CM support";
#endif
}

// Resolve the prefill kernel of one node. Candidate order: preferred GEMM type, the other GEMM type is only
// tried when the preferred one is "cm" (the CM kernels are a comparison aid and are never picked
// automatically), then transcode + oneDNN. With OV_GPU_GGUF_PREFILL_GEMM_STRICT=1, GEMM-family nodes whose
// preferred kernel is unusable fail loudly instead of falling back.
PrefillKernel gguf_select_prefill_kernel(const program_node& node, const RuntimeParams& params, bool shuffled_family, bool family, bool strict) {
    const PrefillKernel preferred = gguf_prefill_gemm_requested();
    const std::string node_name = params.desc ? params.desc->id : std::string("<unknown>");

    const std::string ocl_block = gguf_ocl_gemm_blocker(params, shuffled_family);
    const std::string cm_block = preferred == PrefillKernel::cm ? gguf_cm_gemm_blocker(node, params, shuffled_family) : std::string("not requested");

    const std::string& preferred_block = preferred == PrefillKernel::cm ? cm_block : ocl_block;
    if (preferred_block.empty()) {
        return preferred;
    }
    const char* preferred_name = preferred == PrefillKernel::cm ? "cm" : "ocl";
    if (family && strict) {
        OPENVINO_THROW("[GPU] FCGGUFOpt: node '", node_name, "' cannot use the requested prefill GEMM kernel '", preferred_name, "': ", preferred_block);
    }
    if (preferred == PrefillKernel::cm && ocl_block.empty()) {
        OPENVINO_WARN("[GPU] FCGGUFOpt: node '", node_name, "' cannot use prefill GEMM kernel 'cm' (", cm_block, "); using 'ocl'.");
        return PrefillKernel::ocl;
    }
    if (family) {
        OPENVINO_WARN("[GPU] FCGGUFOpt: node '", node_name, "' prefill falls back to transcode + oneDNN (", preferred_name, " GEMM: ", preferred_block, ").");
    }
    return PrefillKernel::onednn;
}

#ifdef ENABLE_CM_FOR_GPU
// ==========================================================================
// Shared CM kernel-data builder.
//
// The ocl::KernelGenerator base always builds an OCLC_V2 kernel_string loaded
// from the OCL source DB.  CM kernels need language=CM and the CM SourcesDB;
// every CM-native generator in this file (the big GEMM kernels and the small
// f32->f16 activation-conversion kernel) shares that boilerplate here.
// ==========================================================================
class CmKernelGeneratorBase : public KernelGenerator {
public:
    explicit CmKernelGeneratorBase(const char* kernel_name) : KernelGenerator(kernel_name) {}

protected:
    // Override get_kernel_data to produce a CM (not OCLC_V2) kernel descriptor.
    // The base ocl::KernelGenerator::get_kernel_data() always sets language=OCLC_V2
    // and loads templates from the OCL source DB.  For CM kernels we must set
    // language=CM and load from the CM SourcesDB so the kernel compiles correctly.
    [[nodiscard]] KernelData get_kernel_data(const RuntimeParams& params) const override {
        const auto jit = get_jit_constants(params);

        KernelData kd;
        kd.code = std::make_shared<KernelString>();
        kd.code->language = KernelLanguage::CM;
        kd.code->entry_point = get_entry_point(params);
        kd.code->jit = "";
        kd.code->undefs = "";
        kd.code->options = get_build_options(params);
        kd.code->batch_compilation = false;  // Each .cm file has its own #include<cm/cm.h>; do NOT batch
        kd.code->has_microkernels = false;

        // Prepend JIT #define macros, then the .cm kernel source, then #undef.
        // Some jit constants are function-like macros (name contains a
        // parameter list, e.g. "CM_APPLY_POST_OP_0(V, BASE, N)"); #undef only
        // accepts the bare macro identifier, so strip everything from the
        // first '(' onward for the #undef line.
        std::string src;
        for (const auto& c : jit) {
            src += "#define " + c.name + " " + c.value + "\n";
        }
        src += std::string(ov::intel_gpu::cm::SourcesDB::get_kernel_template(get_kernel_name()));
        for (const auto& c : jit) {
            const auto paren = c.name.find('(');
            src += "#undef " + c.name.substr(0, paren) + "\n";
        }
        kd.code->str = std::move(src);

        kd.params.arguments = get_arguments_desc(params);
        kd.params.layerID = params.desc->id;
        kd.update_dispatch_data_func = get_dispatch_data_func();
        kd.need_args_update = true;
        kd.need_dispatch_data_update = true;

        return kd;
    }
};

// ==========================================================================
// CM GEMM kernel generator base.
//
// One subclass per GGUF format so each gets its own kernel binary / cache slot.
// The generator is shape-agnostic with respect to M (passed as scalar at run
// time) and static for K / N (baked into scalars at dispatch time).
//
// Activation is always fp16 by the time this kernel runs: an fp32 activation
// is first converted to a scratch fp16 buffer by convert_act_f32_to_f16.cm
// (see FCGGUFCmImpl::execute_cm_gemm), so this kernel never needs a second
// input ABI.
// ==========================================================================
class FCGGUFCmGemmGenerator : public CmKernelGeneratorBase {
public:
    explicit FCGGUFCmGemmGenerator(const char* kernel_name) : CmKernelGeneratorBase(kernel_name) {}

protected:
    // Build options: -cmc + architecture-specific tile macros.
    // Do NOT call the ocl::KernelGenerator base class get_build_options() here;
    // it returns OCL-specific flags that are incompatible with the CM compiler.
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        const auto& in1 = params.input_layouts[1];
        const size_t K = in1.get_shape()[1];
        const CmGemmConfig cfg = pick_cm_gemm_config(params, K);

        std::string opts = " -cmc";
        opts += " -DROW_GROUPS=" + std::to_string(cfg.row_groups);
        opts += " -DTOKEN_GROUPS=" + std::to_string(cfg.token_groups);
        opts += " -DTOKEN_LOCAL=" + std::to_string(cfg.token_local);
        opts += " -DROW_BLOCKS=" + std::to_string(cfg.row_blocks);
        opts += " -DA_OUTER=" + std::to_string(cfg.a_outer);
        if (params.get_output_layout(0).data_type == data_types::f32)
            opts += " -DCM_OUTPUT_F32=1";
        if (cfg.slm_jblk >= 0)
            opts += " -DSLM_JBLK=" + std::to_string(cfg.slm_jblk);
        if (cfg.pf_a >= 0)
            opts += " -DPF_A=" + std::to_string(cfg.pf_a);
        return opts;
    }

    // The entry point MUST match the literal function name in the .cm source file.
    // Unlike OCL kernels (which use a KERNEL_NAME macro), our CM GEMM kernels use
    // fixed function names.  K and N are passed as runtime scalars, so the same
    // binary serves all shapes — no per-shape suffix needed.
    [[nodiscard]] std::string get_entry_point(const RuntimeParams&) const override {
        return get_kernel_name();  // e.g. "gemm_q41_full" — matches void in the .cm file
    }

    // Generate CM-native post-ops JIT constants from params.fused_desc.
    // We do NOT use the OCL FUSED_OPS / FUSED_OPS_RESULT system because OCL
    // macros contain OCL-specific type conversions (convert_half, __global, etc.)
    // that don't compile in CM.  Instead we emit:
    //   CM_POST_OPS_NUM        — number of post-ops
    //   CM_POST_OPS_DECLS      — extra kernel signature parameters (outer-dep inputs)
    //   CM_APPLY_POST_OP_N(V, BASE, N) — function-like macro, VECTORIZED: it
    //       mutates the WHOLE output vector `V` (width N, base output element
    //       index BASE) in a couple of SIMD instructions.
    //
    // IMPORTANT: this must stay vectorized. An earlier version applied post-ops
    // with a scalar per-lane statement (`v`/`out_idx`) wrapped in a
    // `#pragma unroll for (int _j = 0; _j < N; _j++)` loop at the epilogue.
    // Since the epilogue itself is already unrolled over
    // TOKEN_GROUPS*TOKENS_PER_TILE (=32) token stores, that produced up to
    // ~1024 unrolled scalar float ops PER THREAD, PLUS N scalar
    // (non-coalesced) single-element loads of the fused eltwise operand
    // instead of one contiguous vector load -- this caused a large measured
    // performance regression (bloated code size / register pressure / scalar
    // memory traffic), not classic dpas-loop register spill.
    // Reorder fused ops are layout-only and intentionally skipped.
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        if (!params.has_fused_primitives())
            return {};

        // Collect actionable (non-reorder) fused ops and their apply statements.
        struct PostOp {
            std::string apply_stmt;  // CM code; references vector `V` (float, width N) and uint `BASE`
            bool has_input;          // true if references cm_post_inN pointer
            size_t input_idx;        // index of the outer-dep input pointer (0, 1, ...)
        };
        std::vector<PostOp> ops;
        size_t n_inputs = 0;
        // input_dtypes[input_idx] = actual runtime dtype of that outer-dep memory
        // (fused eltwise operands are NOT always fp16 -- e.g. a residual Add
        // feeding an fp32-output FC node has an fp32 outer dep).
        std::vector<data_types> input_dtypes;

        for (const auto& fd : params.fused_desc) {
            if (fd.is_type<cldnn::reorder>())
                continue;

            PostOp op{};
            op.has_input = fd.has_outer_dep();
            op.input_idx = op.has_input ? n_inputs++ : 0;
            if (op.has_input) {
                input_dtypes.push_back(params.input_layouts[fd.outer_dep_start_idx].data_type);
            }

            if (fd.is_type<cldnn::activation>()) {
                const auto act = fd.typed_desc<cldnn::activation>();
                switch (act->activation_function) {
                case cldnn::activation_func::swish:
                    // swish(x) = x / (1 + exp(-x))
                    // CM has no expf(); use cm_exp(x*log2e) = e^x  (base-2 hardware
                    // exp, vector form -- cm_exp deduces width from its argument).
                    op.apply_stmt = "V = V / (1.0f + cm_exp(-V * 1.4426950408889634f));";
                    break;
                case cldnn::activation_func::relu:
                    op.apply_stmt = "V = cm_max<float>(V, 0.0f);";
                    break;
                case cldnn::activation_func::gelu:
                case cldnn::activation_func::gelu_tanh:
                    // gelu(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
                    // tanh via cm_exp: tanh(a) = (e^2a - 1) / (e^2a + 1)
                    op.apply_stmt = "{ vector<float, N> _ga = 0.7978845608028654f * "
                                    "(V + 0.044715f * V * V * V); "
                                    "vector<float, N> _ge2a = cm_exp(2.0f * _ga * 1.4426950408889634f); "
                                    "V = V * 0.5f * (1.0f + (_ge2a - 1.0f) * cm_inv(_ge2a + 1.0f)); }";
                    break;
                default:
                    op.apply_stmt = "";  // unknown activation — pass-through
                    break;
                }
            } else if (fd.is_type<cldnn::eltwise>() && op.has_input) {
                const auto elt = fd.typed_desc<cldnn::eltwise>();
                const std::string ptr = "cm_post_in" + std::to_string(op.input_idx);
                // Fused operand is contiguous (same layout/strides as `outputs`),
                // so load it as ONE vector instead of N scalar element loads.
                // The load's element type is baked in statically per the
                // operand's known runtime dtype (see input_dtypes above).
                const bool is_f32 =
                    op.input_idx < input_dtypes.size() && input_dtypes[op.input_idx] == data_types::f32;
                std::string load;
                if (is_f32) {
                    load = "vector<float, N> _pin = cm_ptr_load<float, N>((float*)" + ptr + ", (BASE) * 4u);";
                } else {
                    load = "vector<half, N> _pinh = cm_ptr_load<uint, N/2>((uint*)" + ptr +
                           ", (BASE) * 2u).format<half>(); vector<float, N> _pin; _pin = _pinh;";
                }
                switch (elt->mode) {
                case cldnn::eltwise_mode::sum:
                    op.apply_stmt = "{ " + load + " V += _pin; }";
                    break;
                case cldnn::eltwise_mode::prod:
                    op.apply_stmt = "{ " + load + " V *= _pin; }";
                    break;
                default:
                    op.apply_stmt = "";  // unsupported mode — skip
                    break;
                }
            }

            if (!op.apply_stmt.empty())
                ops.push_back(std::move(op));
        }

        if (ops.empty())
            return {};

        JitConstants jit;
        jit.add(make_jit_constant("CM_POST_OPS_NUM", static_cast<int>(ops.size())));

        for (size_t i = 0; i < ops.size(); i++) {
            // Function-like macro: CM_APPLY_POST_OP_i(V, BASE, N) mutates the
            // WHOLE vector V (width N, base output element index BASE).
            jit.add(make_jit_constant("CM_APPLY_POST_OP_" + std::to_string(i) + "(V, BASE, N)",
                                       std::string(ops[i].apply_stmt)));
        }

        // Build the kernel signature extension for outer-dep pointer arguments.
        if (n_inputs > 0) {
            std::string sig;
            for (size_t i = 0; i < n_inputs; i++) {
                // Declare each fused-op pointer with its actual runtime element
                // type; assuming fp16 unconditionally silently misreads any
                // fp32 fused operand (half the bytes, garbage values).
                const char* elem_type = (input_dtypes[i] == data_types::f32) ? "float" : "half";
                sig += ", ";
                sig += elem_type;
                sig += "* cm_post_in" + std::to_string(i) + " [[type(\"svmptr_t\")]]";  // SVM pointer for GPU USM
            }
            jit.add(make_jit_constant("CM_POST_OPS_DECLS", std::string(sig)));
        }

        return jit;
    }

    // Argument binding: INPUT(0)=activation, INPUT(1)=weight, OUTPUT(0)=output,
    // SCALAR(0)=M(token_len), SCALAR(1)=K(input_len), SCALAR(2)=N(output_len),
    // then INPUT_OF_FUSED_PRIMITIVE(j) for each non-reorder fused op with an outer dep.
    //
    // NOTE: INPUT_OF_FUSED_PRIMITIVE uses an index into the runtime's fused_op_inputs array,
    // which is indexed 0..get_fused_mem_count()-1 (sequential, includes ALL fused outer deps).
    // outer_dep_start_idx is the absolute index in the node's full dep list; we must subtract
    // the fused_mem_offset (= min outer_dep_start_idx across all fused deps) to get the slot.
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});   // activation
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});   // weight (shuffled)
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // fp16/fp32 output
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // token_len (M)
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // input_len (K)
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // output_len (N)

        if (!params.has_fused_primitives())
            return args;

        // Compute fused_mem_offset = minimum outer_dep_start_idx across ALL fused ops with outer deps.
        // This is the index where fused dep memories BEGIN in the node's full deps list.
        int fused_mem_offset = -1;
        for (const auto& fd : params.fused_desc) {
            if (fd.outer_dep_start_idx >= 0) {
                if (fused_mem_offset < 0 || fd.outer_dep_start_idx < fused_mem_offset)
                    fused_mem_offset = fd.outer_dep_start_idx;
            }
        }
        if (fused_mem_offset < 0)
            return args;  // no outer deps at all

        // Add outer-dep inputs for non-reorder fused ops.
        // The slot index j = outer_dep_start_idx - fused_mem_offset gives the position in
        // the runtime's fused_op_inputs[] array (which holds all outer dep memories in order).
        for (const auto& fd : params.fused_desc) {
            if (fd.is_type<cldnn::reorder>())
                continue;  // layout-only, skip
            if (fd.has_outer_dep()) {
                const auto slot = static_cast<uint32_t>(fd.outer_dep_start_idx - fused_mem_offset);
                args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, slot});
            }
        }
        return args;
    }

    // Compute global/local work sizes and scalar values from runtime shapes.
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto& in0 = params.get_input_layout(0);
            const auto& in1 = params.get_input_layout(1);
            const size_t N = in1.get_shape()[0];
            const size_t K = in1.get_shape()[1];
            const size_t M = derive_bm(in0.get_shape());

            const CmGemmConfig cfg = pick_cm_gemm_config(params, K);

            const size_t tokens_per_thread = GGUF_CM_TOKENS_PER_TILE * static_cast<size_t>(cfg.token_groups);
            const size_t token_slots = static_cast<size_t>(cfg.token_local) / static_cast<size_t>(cfg.row_blocks);
            const size_t ntiles = (M + tokens_per_thread - 1) / tokens_per_thread;
            // ngroups = N / (OPG * ROW_GROUPS * ROW_BLOCKS)
            const size_t ngroups = N / (GGUF_GEMV_SG_SIZE * static_cast<size_t>(cfg.row_groups) * static_cast<size_t>(cfg.row_blocks));
            const size_t gsize_y = ((ntiles + token_slots - 1) / token_slots) * static_cast<size_t>(cfg.token_local);

            auto& wgs = kd.params.workGroups;
            wgs.global = {ngroups, gsize_y, 1};
            wgs.local = {1, static_cast<size_t>(cfg.token_local), 1};

            // Scalar values: M (uint32), K (uint32), N (uint32).
            kd.params.scalars.clear();
            auto push_u32 = [&](size_t v) {
                scalar_desc sd{};
                sd.t = scalar_desc::Types::UINT32;
                sd.v.u32 = static_cast<uint32_t>(v);
                kd.params.scalars.push_back(sd);
            };
            push_u32(M);
            push_u32(K);
            push_u32(N);
        }};
    }
};

// ==========================================================================
// f32 -> f16 activation conversion kernel generator (convert_act_f32_to_f16.cm).
//
// Explicit-args only: dispatched via FCGGUFCmImpl::execute_stage_explicit(),
// so get_arguments_desc()/get_dispatch_data_func() are left trivial (mirrors
// the pattern used by FCGGUFShuffleGenerator in fc_gguf_shuffle_gen.hpp).
// ==========================================================================
class FCGGUFCmActConvertGenerator : public CmKernelGeneratorBase {
public:
    FCGGUFCmActConvertGenerator() : CmKernelGeneratorBase("convert_act_f32_to_f16") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams&) const override {
        return " -cmc -DCONV_BLOCK=" + std::to_string(CONV_ACT_BLOCK) + " -DCONV_LOCAL=" + std::to_string(CONV_ACT_LOCAL);
    }
    [[nodiscard]] std::string get_entry_point(const RuntimeParams&) const override {
        return get_kernel_name();
    }
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams&) const override {
        return {};
    }
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {}};
    }
};
#endif  // ENABLE_CM_FOR_GPU
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
    // Prefill GEMM stages, indexed by gguf_gemm_fmt_index(): OCL dpas GEMM (default) ...
    std::array<Stage::Ptr, GGUF_GEMM_FORMATS> gemm_stages = {make_stage<FCGGUFGemmGenerator>(kOclGemmKernels[0], kOclGemmFormatDefines[0]),
                                                             make_stage<FCGGUFGemmGenerator>(kOclGemmKernels[1], kOclGemmFormatDefines[1]),
                                                             make_stage<FCGGUFGemmGenerator>(kOclGemmKernels[2], kOclGemmFormatDefines[2]),
                                                             make_stage<FCGGUFGemmGenerator>(kOclGemmKernels[3], kOclGemmFormatDefines[3]),
                                                             make_stage<FCGGUFGemmGenerator>(kOclGemmKernels[4], kOclGemmFormatDefines[4]),
                                                             make_stage<FCGGUFGemmGenerator>(kOclGemmKernels[5], kOclGemmFormatDefines[5])};
#    ifdef ENABLE_CM_FOR_GPU
    // ... and the CM kernels (OV_GPU_GGUF_PREFILL_GEMM=cm, comparison aid) plus their fp32->fp16 activation pre-pass.
    std::array<Stage::Ptr, GGUF_GEMM_FORMATS> cm_gemm_stages = {make_stage<FCGGUFCmGemmGenerator>(kCmGemmKernels[0]),
                                                                make_stage<FCGGUFCmGemmGenerator>(kCmGemmKernels[1]),
                                                                make_stage<FCGGUFCmGemmGenerator>(kCmGemmKernels[2]),
                                                                make_stage<FCGGUFCmGemmGenerator>(kCmGemmKernels[3]),
                                                                make_stage<FCGGUFCmGemmGenerator>(kCmGemmKernels[4]),
                                                                make_stage<FCGGUFCmGemmGenerator>(kCmGemmKernels[5])};
    Stage::Ptr cm_act_conv_stage = make_stage<FCGGUFCmActConvertGenerator>();
    // fp16 copy of an fp32 activation for the CM GEMM (grow-only); bound as INPUT0 only while the CM GEMM runs.
    cldnn::memory::ptr m_act_f16_scratch;
    size_t m_act_f16_scratch_elems = 0;
    bool m_cm_scratch_active = false;
#    endif
#endif

    // Kernel used for prefill (M > threshold) on this node; chosen once in the constructor (see
    // gguf_select_prefill_kernel) and serialized so a cached model behaves the same on load.
    PrefillKernel m_prefill_kernel = PrefillKernel::onednn;
    // OV_GPU_GGUF_PREFILL_GEMM_STRICT: GEMM-family nodes never silently leave their GEMM kernel.
    bool m_gemm_strict = false;
    bool m_gemm_family = false;

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

    // Opt-in f16 prefill (OV_GPU_GGUF_PREFILL_F16=1, default on): the prefill transcode fully
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
        m_prefill_f16 = gguf_prefill_f16_enabled();
        m_use_dq_act = !m_prefill_f16 && gguf_prefill_dq_act_enabled();
    }
    FCGGUFOptImpl(const program_node& node, const RuntimeParams& params) : FCGGUFOptImpl() {
        add_stage(gguf_stage, params);
#ifdef ENABLE_ONEDNN_FOR_GPU
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
        // Pick the prefill kernel of this node and compile only its stages (the transcode / act-DQ kernels
        // are only needed by the oneDNN fallback, the CM kernels only when OV_GPU_GGUF_PREFILL_GEMM=cm).
        const int fmt_idx = gguf_gemm_fmt_index(wet);
        m_gemm_family = fmt_idx >= 0;
        m_gemm_strict = gguf_prefill_gemm_strict();
        m_prefill_kernel = gguf_select_prefill_kernel(node, params, shuffle_applicable && m_gemm_family, m_gemm_family, m_gemm_strict);
        switch (m_prefill_kernel) {
        case PrefillKernel::ocl:
            add_stage(gemm_stages[static_cast<size_t>(fmt_idx)], params);
            break;
        case PrefillKernel::cm:
#    ifdef ENABLE_CM_FOR_GPU
            add_stage(cm_gemm_stages[static_cast<size_t>(fmt_idx)], params);
            add_stage(cm_act_conv_stage, params);
#    endif
            break;
        case PrefillKernel::onednn:
            add_stage(transcode_stage, params);
            add_stage(act_dq_stage, params);
            break;
        case PrefillKernel::none:
            break;
        }
#endif
    }

    void save(cldnn::BinaryOutputBuffer& ob) const override {
        PrimitiveImplOCL::save(ob);
        ob << m_prefill_threshold << m_q4k_shuffle << m_q5k_shuffle << m_q6k_shuffle
           << m_q4_0_shuffle << m_q4_1_shuffle << m_q8_0_shuffle << m_prefill_f16 << m_use_dq_act;
        const auto prefill_kernel = static_cast<uint8_t>(m_prefill_kernel);
        ob << prefill_kernel << m_gemm_strict << m_gemm_family;
    }

    void load(cldnn::BinaryInputBuffer& ib) override {
        PrimitiveImplOCL::load(ib);
        ib >> m_prefill_threshold >> m_q4k_shuffle >> m_q5k_shuffle >> m_q6k_shuffle
           >> m_q4_0_shuffle >> m_q4_1_shuffle >> m_q8_0_shuffle >> m_prefill_f16 >> m_use_dq_act;
        uint8_t prefill_kernel = 0;
        ib >> prefill_kernel >> m_gemm_strict >> m_gemm_family;
        m_prefill_kernel = static_cast<PrefillKernel>(prefill_kernel);
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
        c->m_prefill_kernel = m_prefill_kernel;
        c->m_gemm_strict = m_gemm_strict;
        c->m_gemm_family = m_gemm_family;
        return copy;
    }

    // Bind activation (INPUT0) + weight (INPUT1) + output for the GEMV stage. The empty scale/ZP FC
    // dependencies are intentionally not referenced by that kernel's descriptor.
    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        cldnn::kernel_arguments_data data;
        const auto* fc_inst = dynamic_cast<const cldnn::fully_connected_inst*>(&instance);

#if defined(ENABLE_ONEDNN_FOR_GPU) && defined(ENABLE_CM_FOR_GPU)
        // CM GEMM reads an fp16 copy of an fp32 activation (see execute_cm_gemm); only while that stage runs.
        if (m_cm_scratch_active && m_act_f16_scratch) {
            data.inputs.push_back(m_act_f16_scratch);
        } else {
            data.inputs.push_back(instance.dep_memory_ptr(0));  // activation
        }
#else
        data.inputs.push_back(instance.dep_memory_ptr(0));  // activation
#endif
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
        kernel_dump_info.clear_entries();
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
            bool binary_post_ops_match_output = true;
            for (const auto& fused : params.fused_desc) {
                if (fused.is_type<cldnn::eltwise>()) {
                    if (!fused.has_outer_dep()) {
                        binary_post_ops_match_output = false;
                        break;
                    }
                    const auto& operand = params.get_input_layout(fused.outer_dep_start_idx);
                    const auto& output = params.get_output_layout(0);
                    if (operand.is_dynamic() || operand.get_partial_shape() != output.get_partial_shape() ||
                        operand.format != format::bfyx || operand.data_padding) {
                        binary_post_ops_match_output = false;
                        break;
                    }
                }
            }
            if (M > decode_cutoff && binary_post_ops_match_output) {
                // Prefill: the kernel was chosen per node in the constructor (OCL GEMM by default; CM GEMM
                // for comparison; transcode + oneDNN as the fallback for nodes without a GEMM kernel).
                const size_t fmt = static_cast<size_t>(std::max(0, gguf_gemm_fmt_index(static_cast<element::Type_t>(in1.data_type))));
                switch (m_prefill_kernel) {
                case PrefillKernel::ocl:
                    return execute_ocl_gemm(events, instance, fmt);
                case PrefillKernel::cm:
#    ifdef ENABLE_CM_FOR_GPU
                    return execute_cm_gemm(events, instance, M, fmt);
#    else
                    break;
#    endif
                case PrefillKernel::onednn:
                    return execute_transcode_plus_onednn_woq(events, instance, M);
                case PrefillKernel::none:
                    break;
                }
            }
            if (M > decode_cutoff && !binary_post_ops_match_output && m_gemm_strict && m_gemm_family) {
                OPENVINO_THROW("[GPU] FCGGUFOpt: OV_GPU_GGUF_PREFILL_GEMM_STRICT is set but the fused eltwise operand of node '",
                               params.desc->id,
                               "' does not match the output layout, so the prefill GEMM cannot be used.");
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
                                    bool needs_shape_info = false,
                                    const std::vector<scalar_desc>& scalars = {}) const {
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
        if (!scalars.empty()) {
            for (uint32_t i = 0; i < scalars.size(); i++) {
                desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, i});
            }
            desc.scalars = scalars;
            args.scalars = &desc.scalars;
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

    // ---- Prefill GEMM (OCL): fc_gguf_gemm_opt.cl. Args come from get_arguments() + the stage descriptor. ----
    cldnn::event::ptr execute_ocl_gemm(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance, size_t fmt) {
        Stage& gemm = *gemm_stages[fmt];
        // Dispatch (M tiles) and the activation / output buffers change from call to call.
        gemm.kd.need_dispatch_data_update = true;
        gemm.kd.need_args_update = true;
        return execute_stage(events, instance, gemm);
    }

#    ifdef ENABLE_CM_FOR_GPU
    // ---- Prefill GEMM (CM): gemm_q*_full.cm, kept for comparison (OV_GPU_GGUF_PREFILL_GEMM=cm). ----
    cldnn::event::ptr execute_cm_gemm(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance, size_t M, size_t fmt) {
        Stage& gemm = *cm_gemm_stages[fmt];
        gemm.kd.need_dispatch_data_update = true;
        gemm.kd.need_args_update = true;

        const auto& params = *instance.get_impl_params();
        if (params.get_input_layout(0).data_type != data_types::f32) {
            return execute_stage(events, instance, gemm);
        }

        // fp32 activation: the CM GEMM only reads fp16, so convert into a scratch buffer first and let
        // get_arguments() bind that buffer for the duration of the GEMM.
        const size_t K = params.get_input_layout(1).get_shape()[1];
        cldnn::event::ptr conv_ev = execute_act_f32_to_f16(events, instance, M * K);
        struct ScratchGuard {
            bool& flag;
            explicit ScratchGuard(bool& f) : flag(f) {
                flag = true;
            }
            ~ScratchGuard() {
                flag = false;
            }
        } guard(m_cm_scratch_active);
        return execute_stage({conv_ev}, instance, gemm);
    }

    // Grow-only (re)allocation of the fp16 activation scratch buffer.
    void ensure_act_f16_scratch(cldnn::primitive_inst& instance, size_t total_elems) {
        if (m_act_f16_scratch && m_act_f16_scratch_elems >= total_elems) {
            return;
        }
        auto& engine = instance.get_network().get_engine();
        auto& stream = instance.get_network().get_stream();
        if (m_act_f16_scratch) {
            // An in-flight GEMM from a previous call may still read the old buffer.
            stream.finish();
        }
        const cldnn::layout scratch_layout(ov::Shape{total_elems}, data_types::f16, cldnn::format::bfyx);
        m_act_f16_scratch = engine.allocate_memory(scratch_layout, engine.get_preferred_memory_allocation_type(), /*reset=*/false);
        m_act_f16_scratch_elems = total_elems;
    }

    // fp32 -> fp16 activation conversion pre-pass (convert_act_f32_to_f16.cm).
    cldnn::event::ptr execute_act_f32_to_f16(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance, size_t total_elems) {
        ensure_act_f16_scratch(instance, total_elems);

        const size_t threads = (total_elems + CONV_ACT_BLOCK - 1) / CONV_ACT_BLOCK;
        const size_t global = ((threads + CONV_ACT_LOCAL - 1) / CONV_ACT_LOCAL) * CONV_ACT_LOCAL;

        scalar_desc total_elems_scalar{};
        total_elems_scalar.t = scalar_desc::Types::UINT32;
        total_elems_scalar.v.u32 = static_cast<uint32_t>(total_elems);

        return execute_stage(events,
                             instance,
                             *cm_act_conv_stage,
                             /*inputs=*/{instance.dep_memory_ptr(0)},
                             /*outputs=*/{m_act_f16_scratch},
                             /*global=*/{global, 1, 1},
                             /*local=*/{static_cast<size_t>(CONV_ACT_LOCAL), 1, 1},
                             /*fused_inputs=*/{},
                             /*needs_shape_info=*/false,
                             /*scalars=*/{total_elems_scalar});
    }
#    endif

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
        dnnl::memory::desc act_zp_md;     // activation ZP [M, groups] — u8, asymmetric mode
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
        int ag;    // activation DQ group size
        int aa;    // activation asymmetric
        bool operator==(const MmKey& o) const {
            return et == o.et && m == o.m && k == o.k && n == o.n &&
                   fused_sig == o.fused_sig && f16 == o.f16 && dq == o.dq && asym == o.asym && rg == o.rg && ag == o.ag && aa == o.aa;
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
            h = h * 31 + std::hash<int>()(k.ag);
            h = h * 31 + std::hash<int>()(k.aa);
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
                           int requant_group,
                           int act_group,
                           bool act_asym) {
        MmKey key{static_cast<int>(et), M, K, N, gguf_fused_signature(fused_ops),
                  use_f16 ? 1 : 0, use_dq_act ? 1 : 0, use_asym ? 1 : 0, requant_group, act_group, act_asym ? 1 : 0};
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
            k->src_md = dnnl::memory::desc({M, K}, act_asym ? dnnl::memory::data_type::u8 : dnnl::memory::data_type::s8,
                                            dnnl::memory::format_tag::ab);
            k->wei_md = dnnl::memory::desc({K, N}, w_dt, dnnl::memory::format_tag::ba);
            // Weight scale: per-REQUANT_GROUP x per-N f16.
            k->scale_md = dnnl::memory::desc({K / requant_group, N},
                                             dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
            const int act_groups = K / act_group;
            k->act_scale_md = dnnl::memory::desc({M, act_groups}, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
            attr.set_scales(DNNL_ARG_SRC,     grouped, dnnl::memory::dims{1, act_group},         dnnl::memory::data_type::f16);
            if (act_asym) {
                k->act_zp_md = dnnl::memory::desc({M, act_groups}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::ab);
                attr.set_zero_points(DNNL_ARG_SRC, grouped, dnnl::memory::dims{1, act_group}, dnnl::memory::data_type::u8);
            }
            if (use_asym) {
                // Weight zero point is independent from activation quantization.
                // It is configured below together with the weight scale.
            }
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
        const bool act_asym = gguf_prefill_dq_act_asymmetric();
        // Use the safe group size: if the user-requested group < K AND the weight is
        // asymmetric, the fast gemm_t would require precomputed B-group-sums (needsBGroupSums)
        // that we don't supply, causing a silent ref_t fallback (~100x slower).
        // gguf_act_dq_safe_group_size() detects this and returns K (per-token) instead.
        const size_t act_group = gguf_act_dq_safe_group_size(static_cast<size_t>(K), et, static_cast<size_t>(N));
        validate_gguf_act_dq_config(static_cast<size_t>(K), act_group, act_asym);
        const size_t act_groups = static_cast<size_t>(K) / act_group;

        // Optionally grow and fill activation scratch (INT8/UINT8 [M, K] + scale/ZP [M, groups]).
        cldnn::memory::ptr act_q_scratch;   // INT8 activation
        cldnn::memory::ptr act_s_scratch;   // f16 per-token scale
        cldnn::memory::ptr act_zp_scratch;  // u8 activation zero point
        if (run_dq) {
            const cldnn::layout aq_layout(ov::Shape{static_cast<size_t>(M), static_cast<size_t>(K)},
                                          act_asym ? data_types::u8 : data_types::i8, cldnn::format::bfyx);
            const cldnn::layout as_layout(ov::Shape{static_cast<size_t>(M), act_groups},
                                          data_types::f16, cldnn::format::bfyx);
            const cldnn::layout az_layout(ov::Shape{static_cast<size_t>(M), act_groups},
                                          data_types::u8, cldnn::format::bfyx);
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
                if (act_asym && (!slot.act_zp || slot.act_zp->size() < az_layout.bytes_count())) {
                    if (slot.act_zp) stream.finish();
                    slot.act_zp = engine.allocate_memory(az_layout, alloc_type, /*reset=*/false);
                }
                act_q_scratch = engine.reinterpret_buffer(*slot.act_int8,  aq_layout);
                act_s_scratch = engine.reinterpret_buffer(*slot.act_scale, as_layout);
                if (act_asym)
                    act_zp_scratch = engine.reinterpret_buffer(*slot.act_zp, az_layout);
            }
            // Activation DQ stage: FP16 [M, K] -> INT8 [M, K] + f16 scale [M, 1].
            // Match dynamic_quantize_gpu_opt MODE_PER_TOKEN dispatch: one
            // work-group per token, with tokens in the second grid dimension.
            const size_t dq_global_x = GGUF_ACT_DQ_WG_SIZE;
            execute_stage({transcode_ev},
                          instance,
                          *act_dq_stage,
                          /*inputs=*/ {instance.dep_memory_ptr(0)},
                          /*outputs=*/ act_asym ? std::vector<cldnn::memory::ptr>{act_q_scratch, act_s_scratch, act_zp_scratch}
                                                : std::vector<cldnn::memory::ptr>{act_q_scratch, act_s_scratch},
                          /*global=*/ {dq_global_x * act_groups, static_cast<size_t>(M), 1},
                          /*local=*/  {static_cast<size_t>(GGUF_ACT_DQ_WG_SIZE), 1, 1});
        }

        auto& mm = get_matmul(et, static_cast<int>(M), K, N, src_dt, dst_dt, dnn_engine, fused_ops,
                              use_f16, run_dq, tgt.asymmetric, requant_group, static_cast<int>(act_group), act_asym);

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
            if (act_asym)
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC_0, act_zp_scratch->get_onednn_memory(mm.act_zp_md)});
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
