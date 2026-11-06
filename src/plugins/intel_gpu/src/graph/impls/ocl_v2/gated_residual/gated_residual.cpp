// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_residual.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/gated_residual.hpp"
#include "intel_gpu/primitives/gated_residual_caps.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "registry/implementation_manager.hpp"

namespace ov::intel_gpu::ocl {

using namespace cldnn;
namespace caps = cldnn::gated_residual_caps;

namespace {

constexpr size_t gr_sg = 16;           // subgroup size == one Xe2 hardware thread
constexpr size_t gr_wg_size = 16;      // subgroups of the scalar / single-kernel work-groups
constexpr size_t gr_dpas_k_step = 32;  // K consumed per dpas loop step (two K16 slices)
constexpr size_t gr_dpas_n = 16;       // dpas N axis == GRF width on Xe2
constexpr size_t gr_down_tn = 32;      // low-rank columns per down-projection subgroup
constexpr size_t gr_decode_rows = 32;  // up to here both GEMMs use decode tiling
constexpr size_t gr_read_dpas_tm = 32;
constexpr size_t gr_read_dpas_nt = 2;

size_t get_num_rows(const layout& l, size_t last_dim) {
    const auto count = l.count();
    OPENVINO_ASSERT(last_dim > 0 && count % last_dim == 0, "[GPU][OCL] gated_residual: input element count is not divisible by the last dimension");
    return count / last_dim;
}

/// Widest elements-per-subgroup tile (a multiple of 16) that evenly divides `width`.
size_t gr_best_tile(size_t width) {
    for (size_t t : {size_t{128}, size_t{64}, size_t{32}, size_t{16}}) {
        if (width % t == 0) {
            return t;
        }
    }
    return 16;
}

/// Chunk (halves) of a K loop over `width` elements, at most 64 (one 128 B block message per lane group).
size_t gr_k_chunk(size_t width) {
    return (width % 64 == 0) ? 64 : ((width % 32 == 0) ? 32 : 16);
}

/// gr_read work-group size in subgroups: Xe2+ runs 32 in 256-GRF mode, which halves the per-row latency of the two GEMVs.
size_t gr_read_wg_size(const cldnn::device_info& info) {
    return info.arch >= gpu_arch::xe2 ? 32 : gr_wg_size;
}

/// Token rows per gr_read work-group: every weight chunk is reused for all of them, so as many as the SLM budget (Rbar f16
/// + t f32 per row) allows. Decode keeps one row per work-group.
size_t gr_read_rows(const gated_residual& desc, const cldnn::device_info& info, bool decode) {
    if (decode) {
        return 1;
    }
    const size_t per_row = desc.hc_count * desc.hidden_size * 2 + (desc.lowrank + desc.hc_count + 3) / 4 * 4 * 4;
    const size_t budget = std::min<size_t>(info.max_local_mem_size, 96 * 1024);
    for (size_t rb : {size_t{8}, size_t{4}, size_t{2}}) {
        if (rb * per_row <= budget) {
            return rb;
        }
    }
    return 1;
}

size_t largest_divisor_up_to(size_t n, size_t cap) {
    for (size_t d = std::min(n, cap); d > 1; d--) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

/// Weight storage of the READ / FINAL_MIX kernels (GR_W_TYPE): f16, or the compressed forms with f16 scales and optional zero
/// points. u4 stands for the 4-bit types (i4 / u4, group 32), u8 for the 8-bit ones (i8 / u8, per-channel); signedness is
/// gr_weight_signed().
enum class GrWType : int { f16 = 0, u4 = 1, u8 = 2 };

using GrOp = ov::intel_gpu::op::GatedResidual;

/// Row-count regimes of the two dpas GEMM stages: few rows need the K reduction split over a work-group so the weights are
/// streamed by enough threads; the split stops paying off once the token tiles alone fill the GPU. `gemv` is the decode tier
/// of the compressed DOWN_PROJ for up to gr_gemv_rows rows (one 8-token dpas tile); every other kernel treats it as decode.
enum class GrTier { gemv, decode, mid, wide };

constexpr size_t gr_gemv_rows = 8;

bool gr_decode_like(GrTier tier) {
    return tier == GrTier::gemv || tier == GrTier::decode;
}

GrTier gr_down_tier(size_t num_rows, const cldnn::device_info& info) {
    if (num_rows <= gr_decode_rows) {
        return num_rows <= gr_gemv_rows ? GrTier::gemv : GrTier::decode;
    }
    return num_rows <= 4 * static_cast<size_t>(info.execution_units_count) ? GrTier::mid : GrTier::wide;
}

GrTier gr_up_tier(size_t num_rows) {
    return num_rows <= gr_decode_rows ? GrTier::decode : GrTier::wide;
}

struct GrGemmTiling {
    size_t tm;             // tokens per subgroup
    size_t ksplit;         // subgroups of a work-group splitting the K reduction
    size_t lws_y;          // token tiles per work-group (ksplit == 1 only)
    size_t wq_bytes = 32;  // compressed per-subgroup kernels: per-lane weight bytes per load (see gr_wq_kch)
};

GrGemmTiling gr_down_tiling(const gated_residual& desc, const cldnn::device_info& info, GrTier tier, GrWType wtype = GrWType::f16) {
    const size_t cd_steps = desc.hc_count * desc.hidden_size / gr_dpas_k_step;
    const bool compressed = wtype != GrWType::f16;
    // Dequantizing makes every K step longer, so compressed weights want more subgroups splitting K.
    const size_t boost = compressed ? 2 : 1;
    switch (tier) {
    case GrTier::gemv:
    case GrTier::decode: {
        // the compressed gemv tier uses one 8-token tile per subgroup, which halves the unpack / dpas work
        const size_t tm = compressed && tier == GrTier::gemv ? size_t{8} : size_t{16};
        const size_t ksplit = largest_divisor_up_to(cd_steps, 8 * boost);
        if (tier == GrTier::gemv && wtype == GrWType::u4) {
            // Wide 4-bit DOWN_PROJs whose split fills more than 3/4 of the resident subgroups (4 per EU with 256 GRF) stream
            // faster as fewer, longer K segments read 64 B per lane, filling at most 2/3 of them in one wave.
            const size_t wgs = desc.lowrank / gr_dpas_n + (desc.produce_write_gate ? 1 : 0);
            const size_t resident = 4 * static_cast<size_t>(info.execution_units_count);
            if (4 * wgs * ksplit > 3 * resident) {
                return {tm, largest_divisor_up_to(cd_steps, std::max<size_t>(1, 2 * resident / (3 * wgs))), 1, 64};
            }
        }
        return {tm, ksplit, 1};
    }
    case GrTier::mid:
        return {32, largest_divisor_up_to(cd_steps, (info.execution_units_count >= 128 ? 4 : 2) * boost), 1};
    default:
        return {32, 1, 1};
    }
}

GrGemmTiling gr_up_tiling(const gated_residual& desc, const cldnn::device_info& info, GrTier tier, GrWType wtype) {
    if (!gr_decode_like(tier)) {
        return {16, 1, 8};
    }
    // Narrow D leaves few column tiles; split L until there are ~2 subgroups per EU. Compressed weights move fewer bytes per
    // K step and split further: u4 down to two K32 steps per subgroup, u8 down to one.
    const bool compressed = wtype != GrWType::f16;
    const size_t col_tiles = desc.hidden_size / gr_dpas_n;
    const size_t eu = info.execution_units_count;
    const size_t steps = desc.lowrank / gr_dpas_k_step;
    const size_t cap = wtype == GrWType::u8 ? steps : compressed ? std::min((8 * eu + col_tiles - 1) / col_tiles, steps / 2) : 2 * eu / col_tiles;
    // f16 reads every W_up byte once per token tile, so a 16-token tile halves that traffic; the compressed kernels unpack the
    // weights again per tile and keep 8.
    return {compressed ? size_t{8} : size_t{16}, largest_divisor_up_to(steps, std::max<size_t>(1, cap)), 1};
}

const char* gr_tier_suffix(GrTier tier, bool down, bool q = false, bool pd = false) {
    if (pd) {
        switch (tier) {
        case GrTier::decode:
            return down ? "down_pd_decode" : "up_pd_decode";
        case GrTier::mid:
            return "down_pd_mid";
        default:
            return down ? "down_pd" : "up_pd";
        }
    }
    if (q) {
        switch (tier) {
        case GrTier::gemv:
            return "down_q_gemv";
        case GrTier::decode:
            return down ? "down_q_decode" : "up_q_decode";
        case GrTier::mid:
            return "down_q_mid";
        default:
            return down ? "down_q" : "up_q";
        }
    }
    switch (tier) {
    case GrTier::decode:
        return down ? "down_decode" : "up_decode";
    case GrTier::mid:
        return "down_mid";
    default:
        return down ? "down" : "up";
    }
}

/// Above ~4 rows per EU the single systolic kernel gr_read_dpas beats the 3-dispatch pipeline and needs no scratch; below it
/// the pipeline's split-K tiers parallelize better.
bool use_single_dpas_read(const gated_residual& desc, const cldnn::device_info& info, size_t num_rows) {
    return desc.hidden_size % 32 == 0 && num_rows > 4 * static_cast<size_t>(info.execution_units_count);
}

/// Whether the READ / FINAL_MIX runs on the systolic kernels instead of the fused scalar gr_read kernel.
bool can_use_dpas_gr_read(const RuntimeParams& params) {
    return caps::uses_dpas_read(params.typed_desc<gated_residual>()->lowrank, caps::has_systolic(params.get_device_info(), params.get_program().get_config()));
}

GrWType gr_weight_type(const RuntimeParams& params) {
    const auto dt = params.get_input_layout(2).data_type;
    if (dt == ov::element::u4 || dt == ov::element::i4) {
        return GrWType::u4;
    }
    return dt == ov::element::u8 || dt == ov::element::i8 ? GrWType::u8 : GrWType::f16;
}

bool gr_weight_signed(const RuntimeParams& params) {
    const auto dt = params.get_input_layout(2).data_type;
    return dt == ov::element::i4 || dt == ov::element::i8;
}

/// Input form of a READ / FINAL_MIX node (the op validated that its input count is one of the forms).
GrOp::ReadLayout gr_read_layout(const RuntimeParams& params) {
    const auto layout = GrOp::read_layout(params.get_input_layout_size(), params.typed_desc<gated_residual>()->produce_write_gate);
    OPENVINO_ASSERT(layout.has_value(), "[GPU][OCL] gated_residual: unexpected number of inputs ", params.get_input_layout_size());
    return *layout;
}

/// The kernels always take a zp argument; without zero points the scale buffer is bound instead and never read (GR_HAS_ZP == 0).
uint32_t gr_zp_input(const GrOp::ReadLayout& layout, size_t zp_idx, size_t scale_idx) {
    return static_cast<uint32_t>(layout.has_zp ? zp_idx : scale_idx);
}

size_t gr_weight_bits(GrWType t) {
    return t == GrWType::u4 ? 4 : 8;
}

/// K values one lane loads per step from its weight row (K32 slices are dequantized one by one out of the raw registers).
/// Wider loads fill whole cache lines (u4 reads only 16 B per K32 otherwise) but cost registers, so `max_bytes` is chosen per
/// kernel; the width must divide the K range a subgroup walks and keep every load inside one aligned block of the row.
size_t gr_wq_kch(GrWType t, size_t segment, size_t row_k, size_t max_bytes) {
    const size_t bits = gr_weight_bits(t);
    for (size_t bytes : {size_t{64}, size_t{32}, size_t{16}}) {
        const size_t kch = bytes * 8 / bits;
        if (bytes <= max_bytes && kch >= 32 && segment % kch == 0 && row_k % kch == 0) {
            return kch;
        }
    }
    return 32;
}

constexpr size_t gr_wq_up_load_bytes = 64;    // UP: a lane loads up to one cache line of its row per step
constexpr size_t gr_wq_dpas_load_bytes = 32;  // gr_read_dpas is register bound

/// Compressed READ / FINAL_MIX above the decode regime: `st` token subgroups (and `ks` K subgroups) of one work-group dequantize
/// every weight slice once into SLM and all of them run their dpas on it, instead of each token subgroup dequantizing the
/// weights again. The decode tier keeps the per-subgroup dequantization (one or two token tiles, nothing to share).
struct GrQShare {
    size_t st;  // token subgroups per work-group
    size_t ks;  // K subgroups per work-group
};

GrQShare gr_down_q_share(const gated_residual& desc, GrTier tier) {
    const size_t cd_steps = desc.hc_count * desc.hidden_size / gr_dpas_k_step;
    return tier == GrTier::mid ? GrQShare{4, largest_divisor_up_to(cd_steps, 4)} : GrQShare{8, 1};
}

constexpr size_t gr_up_q_st = 4;

/// The 3-dispatch pipeline needs rows * C*D * 2 bytes of Rbar scratch; beyond this the compressed single kernel (which keeps
/// Rbar in SLM, but dequantizes per token tile) is used instead.
size_t gr_q_scratch_limit(const RuntimeParams& params) {
    return GPU_DEBUG_VALUE_OR(params.get_program().get_config().get_gr_q_scratch_limit_mb(), size_t{256}) << 20;
}

bool use_single_read(const RuntimeParams& params, size_t rows, bool compressed);

/// 4-bit decode GEMMs unpack their weights in a permuted K order and read Rbar / T stored with the same permutation (GR_KP). The
/// decode RBAR cannot know the row count of a dynamic shape, so this holds only when no decode row count can take the
/// predequantized or single-kernel path.
bool gr_kperm(const RuntimeParams& params, GrTier tier, bool q, bool pd);

/// Compressed READ above this many rows dequantizes both matrices into f16 scratch buffers with a separate kernel and then runs
/// the f16 kernels; below it the compute kernels dequantize on the fly.
size_t gr_pd_min_rows(const RuntimeParams& params) {
    const int64_t rows = GPU_DEBUG_VALUE_OR(params.get_program().get_config().get_gr_pd_min_rows(), int64_t{-1});
    // Below ~4 rows per EU the dequantizing GEMM kernels win; above ~5 rows per EU a separate dequantization followed by the f16
    // single kernel (no Rbar / T scratch traffic, weights hot in cache) wins.
    return rows >= 0 ? static_cast<size_t>(rows) : 5 * static_cast<size_t>(params.get_device_info().execution_units_count);
}

/// Group size along K (K / G) of the weight whose f16 scale is input `scale_idx`.
size_t gr_group_size(const RuntimeParams& params, size_t scale_idx, size_t k) {
    const auto& ps = params.get_input_layout(scale_idx).get_partial_shape();
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 2 && ps[0].is_static(), "[GPU][OCL] gated_residual: the scale must be a static [G, N] tensor");
    return caps::derive_group_size(k, static_cast<size_t>(ps[0].get_length()));
}

/// GR_<prefix>_KCH / _NRAW / _RB / _GS32 (group size in K32 slices) / _PERCH (one group) of one weight matrix [N, K].
void add_wq_matrix_constants(JitConstants& jit, const std::string& prefix, GrWType t, size_t kch, size_t row_k, size_t group_size) {
    jit.make(prefix + "_KCH", kch);
    jit.make(prefix + "_NRAW", kch * gr_weight_bits(t) / 32);
    jit.make(prefix + "_RB", row_k * gr_weight_bits(t) / 8);
    jit.make(prefix + "_GS32", group_size / gr_dpas_k_step);
    jit.make(prefix + "_PERCH", group_size == row_k ? 1 : 0);
}

/// Weight format of a compressed kernel: bit width class, signedness and whether there are zero points.
void add_wq_format_constants(JitConstants& jit, const RuntimeParams& params) {
    jit.make("GR_W_TYPE", static_cast<int>(gr_weight_type(params)));
    jit.make("GR_W_SIGNED", gr_weight_signed(params) ? 1 : 0);
    jit.make("GR_HAS_ZP", gr_read_layout(params).has_zp ? 1 : 0);
}

/// Float as an OpenCL C literal with full single precision (an unsuffixed literal would be a double).
JitConstant float_constant(const std::string& name, float value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9gf", static_cast<double>(value));
    return JitConstant(name, buf);
}

void push_num_rows(KernelData& kd, size_t num_rows) {
    kd.params.scalars.clear();
    scalar_desc s;
    s.t = scalar_desc::Types::UINT32;
    s.v.u32 = static_cast<uint32_t>(num_rows);
    kd.params.scalars.push_back(s);
}

size_t rows_of(const RuntimeParams& params, const gated_residual& desc) {
    return get_num_rows(params.get_input_layout(0), desc.hc_count * desc.hidden_size);
}

/// Common base of the stages: the kernels take their geometry from JIT constants and `num_rows`, never from shape_info or
/// the INPUT0_* layout macros, so only the base JIT constants are used.
class GrGenerator : public KernelGenerator {
public:
    GrGenerator(std::string_view template_name, std::string_view suffix, bool large_grf = true)
        : KernelGenerator(template_name, suffix),
          m_large_grf(large_grf) {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        auto options = KernelGenerator::get_build_options(params);
        const bool large = m_large_grf;
        if (large && params.get_device_info().arch >= gpu_arch::xe_hpg) {
            options += " -cl-intel-256-GRF-per-thread";
        }
        return options;
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        jit.make("SUBGROUP_SIZE", gr_sg);
        jit.make("GR_BRANCHES", desc->hc_count);
        jit.make("GR_HIDDEN", desc->hidden_size);
        return jit;
    }

    void add_norm_constants(JitConstants& jit, const gated_residual& desc) const {
        jit.add(float_constant("GR_EPS", desc.epsilon));
        jit.make("GR_NORM_GAIN_PLUS_ONE", desc.norm_gain_plus_one ? 1 : 0);
    }

    /// The single-dispatch READ kernels have the write-gate column tile for f16 weights only. A compressed READ computes it in
    /// the decode DOWN_PROJ tile, in a separate GATE stage, or, when the weights are dequantized into f16 scratch (`pd`), in the
    /// f16 kernel on the dequantized W_inject.
    [[nodiscard]] static bool gate_in_kernel(const gated_residual& desc, GrWType wtype) {
        return desc.produce_write_gate && wtype == GrWType::f16;
    }

    /// `gate_all_types`: the kernel computes the write gate for compressed weights too (the scalar kernel; the dpas kernels have the
    /// write-gate tile for f16 weights only).
    [[nodiscard]] static Arguments read_arguments(const gated_residual& desc,
                                                  const GrOp::ReadLayout& layout,
                                                  GrWType wtype,
                                                  bool pd = false,
                                                  bool gate_all_types = false) {
        const bool gate = desc.produce_write_gate && (wtype == GrWType::f16 || gate_all_types);
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // residual
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // norm weight
        if (pd) {
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GatedResidualInternalBufIdx::WD_F16)});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GatedResidualInternalBufIdx::WU_F16)});
        } else {
            args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // W_down
            args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // W_up
        }
        if (wtype != GrWType::f16) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(GrOp::ReadInputIdx::SCALE_DOWN)});  // [G, L]
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(GrOp::ReadInputIdx::SCALE_UP)});    // [G, C*D]
            args.push_back({ArgumentDescriptor::Types::INPUT, gr_zp_input(layout, GrOp::ReadInputIdx::ZP_DOWN, GrOp::ReadInputIdx::SCALE_DOWN)});
            args.push_back({ArgumentDescriptor::Types::INPUT, gr_zp_input(layout, GrOp::ReadInputIdx::ZP_UP, GrOp::ReadInputIdx::SCALE_UP)});
        }
        if (gate) {
            if (pd) {
                args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GatedResidualInternalBufIdx::WI_F16)});
            } else {
                args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(layout.w_inject)});
                if (wtype != GrWType::f16) {
                    args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(layout.scale_inject)});
                    args.push_back({ArgumentDescriptor::Types::INPUT, gr_zp_input(layout, layout.zp_inject, layout.scale_inject)});
                }
            }
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // mixed
        if (gate) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});  // write gate S
        }
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // num_rows
        return args;
    }

private:
    bool m_large_grf;
};

// ---------------------------------------------------------------------------
// READ / MIX, fused scalar kernel
// ---------------------------------------------------------------------------
class GrReadGenerator : public GrGenerator {
public:
    explicit GrReadGenerator(bool decode = false) : GrGenerator("gr_read", decode ? "decode" : "wide"), m_decode(decode) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GrGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        add_norm_constants(jit, *desc);
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.make("GR_PRODUCE_WRITE_GATE", desc->produce_write_gate ? 1 : 0);
        jit.make("GR_WG_SIZE", gr_read_wg_size(params.get_device_info()));
        jit.make("GR_READ_ROWS", gr_read_rows(*desc, params.get_device_info(), m_decode));
        jit.make("GR_NV_D", gr_k_chunk(desc->hidden_size) / 16);
        jit.make("GR_NV_C", gr_k_chunk(desc->hc_count * desc->hidden_size) / 16);
        jit.make("GR_NV_L", gr_k_chunk(desc->lowrank) / 16);
        if (gr_weight_type(params) != GrWType::f16) {
            const size_t cd = desc->hc_count * desc->hidden_size;
            add_wq_format_constants(jit, params);
            jit.make("GR_RD_GSD", gr_group_size(params, GrOp::ReadInputIdx::SCALE_DOWN, cd));
            jit.make("GR_RD_GSU", gr_group_size(params, GrOp::ReadInputIdx::SCALE_UP, desc->lowrank));
            if (desc->produce_write_gate) {
                jit.make("GR_RD_GSI", gr_group_size(params, gr_read_layout(params).scale_inject, cd));
            }
        }
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return read_arguments(*params.typed_desc<gated_residual>(), gr_read_layout(params), gr_weight_type(params), false, /*gate_all_types=*/true);
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[decode = m_decode](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = rows_of(params, *desc);
            const size_t wg = gr_read_wg_size(params.get_device_info());
            const size_t rows_per_wg = gr_read_rows(*desc, params.get_device_info(), decode);

            auto& wgs = kd.params.workGroups;
            wgs.global = {wg * gr_sg, std::max<size_t>(1, (num_rows + rows_per_wg - 1) / rows_per_wg), 1};
            wgs.local = {wg * gr_sg, 1, 1};
            push_num_rows(kd, num_rows);
        }};
    }

private:
    bool m_decode;
};

// ---------------------------------------------------------------------------
// READ / MIX, single systolic kernel (many rows)
// ---------------------------------------------------------------------------
class GrReadDpasGenerator : public GrGenerator {
public:
    explicit GrReadDpasGenerator(bool q = false, bool pd = false) : GrGenerator("gr_read_dpas", q ? "q" : (pd ? "pd" : "")), m_q(q), m_pd(pd) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GrGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        add_norm_constants(jit, *desc);
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.make("GR_PRODUCE_WRITE_GATE", desc->produce_write_gate && !m_q ? 1 : 0);
        jit.make("GR_WG_SIZE", gr_wg_size);
        jit.make("GR_RD_TM", gr_read_dpas_tm);
        jit.make("GR_RD_NT", gr_read_dpas_nt);
        jit.make("GR_NV_KD", (desc->hidden_size % 128 == 0 ? 128 : 32) / 16);
        const size_t cd = desc->hc_count * desc->hidden_size;
        const size_t slm_bytes = desc->hc_count * gr_read_dpas_tm * 4 + gr_read_dpas_tm * desc->lowrank * 2;
        // Larger SLM footprints shrink L1, which phase 2 relies on to share the residual tile between subgroups.
        const size_t slm_limit = std::min<size_t>(params.get_device_info().max_local_mem_size, 64 * 1024);
        const int slm_gain = slm_bytes + cd * 4 <= slm_limit ? 2 : (slm_bytes + (cd + 48) * 2 <= slm_limit ? 1 : 0);
        jit.make("GR_RD_SLM_GAIN", slm_gain);
        if (m_q) {
            const auto wtype = gr_weight_type(params);
            add_wq_format_constants(jit, params);
            add_wq_matrix_constants(jit,
                                    "GR_WD",
                                    wtype,
                                    gr_wq_kch(wtype, desc->hidden_size, cd, gr_wq_dpas_load_bytes),
                                    cd,
                                    gr_group_size(params, GrOp::ReadInputIdx::SCALE_DOWN, cd));
            add_wq_matrix_constants(jit,
                                    "GR_WU",
                                    wtype,
                                    gr_wq_kch(wtype, desc->lowrank, desc->lowrank, gr_wq_dpas_load_bytes),
                                    desc->lowrank,
                                    gr_group_size(params, GrOp::ReadInputIdx::SCALE_UP, desc->lowrank));
        }
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return read_arguments(*params.typed_desc<gated_residual>(), gr_read_layout(params), m_q ? gr_weight_type(params) : GrWType::f16, m_pd);
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = rows_of(params, *desc);
            auto& wgs = kd.params.workGroups;
            wgs.global = {gr_wg_size * gr_sg, std::max<size_t>(1, (num_rows + gr_read_dpas_tm - 1) / gr_read_dpas_tm), 1};
            wgs.local = {gr_wg_size * gr_sg, 1, 1};
            push_num_rows(kd, num_rows);
        }};
    }

private:
    bool m_q;
    bool m_pd;
};

// ---------------------------------------------------------------------------
// WRITE / COMBINE
// ---------------------------------------------------------------------------
class GrWriteGenerator : public GrGenerator {
public:
    GrWriteGenerator() : GrGenerator("gr_write", "", /*large_grf=*/false) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GrGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        const size_t tile = gr_best_tile(desc->hidden_size);
        jit.make("GR_WRITE_TILE", tile);
        jit.make("GR_WRITE_NV", tile / 16);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});   // residual
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});   // sub-layer output Y
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});   // write gate S
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // residual'
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // num_rows
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = rows_of(params, *desc);
            const size_t tile = gr_best_tile(desc->hidden_size);
            auto& wgs = kd.params.workGroups;
            wgs.global = {(desc->hidden_size + tile - 1) / tile * gr_sg, num_rows, 1};
            wgs.local = {gr_sg, 1, 1};
            push_num_rows(kd, num_rows);
        }};
    }
};

// ---------------------------------------------------------------------------
// 3-dispatch systolic pipeline: all three stages compile gr_read_pipeline.cl with a different GR_PIPELINE_STAGE
// ---------------------------------------------------------------------------
class GrRbarGenerator : public GrGenerator {
public:
    explicit GrRbarGenerator(bool decode = false) : GrGenerator("gr_read_pipeline", decode ? "rbar_decode" : "rbar"), m_decode(decode) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GrGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        const size_t ch = gr_best_tile(desc->hidden_size);
        jit.make("GR_PIPELINE_STAGE", 0);
        add_norm_constants(jit, *desc);
        jit.make("GR_CH", ch);
        jit.make("GR_NV_CH", ch / 16);
        const size_t nch = desc->hidden_size / ch;
        const size_t s = rbar_split(*desc, m_decode);
        jit.make("GR_NCH", nch);
        jit.make("GR_RB_S", s);
        // Gain chunks are independent of the norm: preload up to 16 (64 GRFs) while the row is still being reduced.
        jit.make("GR_GPF", desc->hidden_size <= 2560 ? std::min<size_t>(nch / s, 16) : 0);
        jit.make("GR_KPERM", m_decode && gr_kperm(params, GrTier::decode, true, false) ? 1 : 0);
        return jit;
    }

    /// Subgroups sharing one branch row. Decode has a work-group per row and little else to run, so the row is split down to
    /// 4 chunks per subgroup; with more rows the SLM reduction only adds latency.
    static size_t rbar_split(const gated_residual& desc, bool decode) {
        const size_t nch = desc.hidden_size / gr_best_tile(desc.hidden_size);
        if (!decode) {
            return 1;
        }
        return largest_divisor_up_to(nch, std::min(std::max<size_t>(1, nch / 4), std::max<size_t>(1, 32 / desc.hc_count)));
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // residual
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // norm weight
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::RBAR});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // num_rows
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[decode = m_decode](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = rows_of(params, *desc);
            auto& wgs = kd.params.workGroups;
            // S subgroups per (branch, row) reduce the branch norm through SLM; one work-group per row.
            const size_t s = rbar_split(*desc, decode);
            wgs.global = {desc->hc_count * s * gr_sg, std::max<size_t>(1, num_rows), 1};
            wgs.local = {desc->hc_count * s * gr_sg, 1, 1};
            push_num_rows(kd, num_rows);
        }};
    }

private:
    bool m_decode;
};

class GrDownProjGenerator : public GrGenerator {
public:
    explicit GrDownProjGenerator(GrTier tier, bool q = false, bool pd = false)
        : GrGenerator("gr_read_pipeline", gr_tier_suffix(tier, true, q, pd)),
          m_tier(tier),
          m_q(q),
          m_pd(pd) {}

    /// The write gate is an extra column tile of every DOWN_PROJ except the work-group shared compressed one.
    static bool gate_tile(const gated_residual& desc, bool q, GrTier tier) {
        return desc.produce_write_gate && (!q || gr_decode_like(tier));
    }

    /// Compressed decode streams 4-8x fewer weight bytes per column tile, so 16-column tiles are used to spread the work over
    /// more work-groups. With two token tiles (9-32 rows) the extra write-gate tile would start a second wave of work-groups,
    /// so that case keeps 32 columns.
    static size_t down_tn(const gated_residual& desc, GrTier tier, bool q, bool share) {
        if (!q || !gr_decode_like(tier) || share || (tier == GrTier::decode && desc.produce_write_gate)) {
            return gr_down_tn;
        }
        return gr_dpas_n;
    }

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GrGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        const auto wtype = m_q ? gr_weight_type(params) : GrWType::f16;
        const bool share = m_q && !gr_decode_like(m_tier);
        const auto qs = gr_down_q_share(*desc, m_tier);
        const auto tiling = share ? GrGemmTiling{32, qs.ks, 1} : gr_down_tiling(*desc, params.get_device_info(), m_tier, wtype);
        jit.make("GR_PIPELINE_STAGE", 1);
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.make("GR_DOWN_TM", tiling.tm);
        jit.make("GR_DOWN_TN", down_tn(*desc, m_tier, m_q, share));
        jit.make("GR_DOWN_KSPLIT", tiling.ksplit);
        jit.make("GR_PRODUCE_WRITE_GATE", gate_tile(*desc, m_q, m_tier) ? 1 : 0);
        if (m_q) {
            const size_t cd = desc->hc_count * desc->hidden_size;
            const auto layout = gr_read_layout(params);
            const size_t gs_down = gr_group_size(params, GrOp::ReadInputIdx::SCALE_DOWN, cd);
            const bool gate = gate_tile(*desc, m_q, m_tier);
            const size_t gs_inject = gate ? gr_group_size(params, layout.scale_inject, cd) : cd;
            add_wq_format_constants(jit, params);
            jit.make("GR_Q_SHARE", share ? 1 : 0);
            jit.make("GR_KPERM", gr_kperm(params, m_tier, m_q, m_pd) ? 1 : 0);
            if (share) {
                jit.make("GR_Q_ST", qs.st);
            }
            // per-channel 8-bit weights (also in the write-gate tile) fold their zero point and scale into the epilogue
            jit.make("GR_WQ_FOLD", !share && wtype == GrWType::u8 && gs_down == cd && gs_inject == cd ? 1 : 0);
            if (gate) {
                jit.make("GR_WI_GS32", gs_inject / gr_dpas_k_step);
            }
            add_wq_matrix_constants(jit, "GR_WD", wtype, share ? 32 : gr_wq_kch(wtype, cd / tiling.ksplit, cd, tiling.wq_bytes), cd, gs_down);
        }
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        const auto desc = params.typed_desc<gated_residual>();
        const auto layout = gr_read_layout(params);
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::RBAR});
        if (m_pd) {
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GatedResidualInternalBufIdx::WD_F16)});
        } else {
            args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // W_down
        }
        if (m_q) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(GrOp::ReadInputIdx::SCALE_DOWN)});
            args.push_back({ArgumentDescriptor::Types::INPUT, gr_zp_input(layout, GrOp::ReadInputIdx::ZP_DOWN, GrOp::ReadInputIdx::SCALE_DOWN)});
        }
        const bool gate = gate_tile(*desc, m_q, m_tier);
        if (gate) {
            if (m_q) {
                args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(layout.w_inject)});
                args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(layout.scale_inject)});
                args.push_back({ArgumentDescriptor::Types::INPUT, gr_zp_input(layout, layout.zp_inject, layout.scale_inject)});
            } else if (m_pd) {
                args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GatedResidualInternalBufIdx::WI_F16)});
            } else {
                args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(layout.w_inject)});
            }
        }
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::T_SCRATCH});
        if (gate) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});  // write gate S
        }
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // num_rows
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[tier = m_tier, q = m_q](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = rows_of(params, *desc);
            const bool share = q && !gr_decode_like(tier);
            const auto qs = gr_down_q_share(*desc, tier);
            const auto tiling =
                share ? GrGemmTiling{32, qs.ks, 1} : gr_down_tiling(*desc, params.get_device_info(), tier, q ? gr_weight_type(params) : GrWType::f16);
            const size_t col_tiles = desc->lowrank / down_tn(*desc, tier, q, share) + (gate_tile(*desc, q, tier) ? 1 : 0);
            const size_t row_tiles = std::max<size_t>(1, (num_rows + tiling.tm - 1) / tiling.tm);

            auto& wgs = kd.params.workGroups;
            if (share) {
                // one work-group = st token subgroups x ks K subgroups on one column tile
                wgs.global = {col_tiles * qs.st * qs.ks * gr_sg, (row_tiles + qs.st - 1) / qs.st, 1};
                wgs.local = {qs.st * qs.ks * gr_sg, 1, 1};
            } else {
                // One work-group per (column tile, token tile); KSPLIT subgroups split the K reduction.
                wgs.global = {col_tiles * tiling.ksplit * gr_sg, row_tiles, 1};
                wgs.local = {tiling.ksplit * gr_sg, 1, 1};
            }
            push_num_rows(kd, num_rows);
        }};
    }

private:
    GrTier m_tier;
    bool m_q;
    bool m_pd;
};

const char* gr_gate_suffix(GrTier tier) {
    return tier == GrTier::mid ? "gate_mid" : "gate";
}

/// Write gate S = 2 * sigmoid(Rbar . W_inject^T / C) of a compressed READ above the decode regime: the f16 DOWN_PROJ kernel
/// restricted to its write-gate column tile, reading the Rbar scratch and the dequantized W_inject (the work-group shared
/// compressed DOWN_PROJ has no such tile; the decode one has).
class GrGateProjGenerator : public GrGenerator {
public:
    explicit GrGateProjGenerator(GrTier tier) : GrGenerator("gr_read_pipeline", gr_gate_suffix(tier)), m_tier(tier) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GrGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        const auto tiling = gr_down_tiling(*desc, params.get_device_info(), m_tier);
        jit.make("GR_PIPELINE_STAGE", 1);
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.make("GR_DOWN_TM", tiling.tm);
        jit.make("GR_DOWN_TN", gr_dpas_n);
        jit.make("GR_DOWN_KSPLIT", tiling.ksplit);
        jit.make("GR_PRODUCE_WRITE_GATE", 1);
        jit.make("GR_GATE_ONLY", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::RBAR});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GatedResidualInternalBufIdx::WI_F16)});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});  // write gate S
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // num_rows
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[tier = m_tier](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = rows_of(params, *desc);
            const auto tiling = gr_down_tiling(*desc, params.get_device_info(), tier);
            const size_t row_tiles = std::max<size_t>(1, (num_rows + tiling.tm - 1) / tiling.tm);
            auto& wgs = kd.params.workGroups;
            wgs.global = {tiling.ksplit * gr_sg, row_tiles, 1};
            wgs.local = {tiling.ksplit * gr_sg, 1, 1};
            push_num_rows(kd, num_rows);
        }};
    }

private:
    GrTier m_tier;
};

class GrUpProjGenerator : public GrGenerator {
public:
    explicit GrUpProjGenerator(GrTier tier, bool q = false, bool pd = false)
        : GrGenerator("gr_read_pipeline", gr_tier_suffix(tier, false, q, pd)),
          m_tier(tier),
          m_q(q),
          m_pd(pd) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GrGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        const bool share = m_q && !gr_decode_like(m_tier);
        // the 4-bit decode (K-permuted) runs one subgroup per branch over the whole low rank
        const bool bsplit = gr_kperm(params, m_tier, m_q, m_pd);
        const auto tiling = share    ? GrGemmTiling{16, 1, 1}
                            : bsplit ? GrGemmTiling{8, 1, 1}
                                     : gr_up_tiling(*desc, params.get_device_info(), m_tier, m_q ? gr_weight_type(params) : GrWType::f16);
        jit.make("GR_PIPELINE_STAGE", 2);
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.make("GR_UP_TM", tiling.tm);
        jit.make("GR_UP_KSPLIT", tiling.ksplit);
        jit.make("GR_UP_BSPLIT", bsplit ? 1 : 0);
        if (m_q) {
            const auto wtype = gr_weight_type(params);
            const size_t gs_up = gr_group_size(params, GrOp::ReadInputIdx::SCALE_UP, desc->lowrank);
            add_wq_format_constants(jit, params);
            jit.make("GR_Q_SHARE", share ? 1 : 0);
            jit.make("GR_KPERM", gr_kperm(params, m_tier, m_q, m_pd) ? 1 : 0);
            if (share) {
                jit.make("GR_Q_ST", gr_up_q_st);
            }
            jit.make("GR_WQ_FOLD", !share && wtype == GrWType::u8 && gs_up == desc->lowrank ? 1 : 0);
            add_wq_matrix_constants(jit,
                                    "GR_WU",
                                    wtype,
                                    share ? 32 : gr_wq_kch(wtype, desc->lowrank / tiling.ksplit, desc->lowrank, gr_wq_up_load_bytes),
                                    desc->lowrank,
                                    gs_up);
        }
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        const auto layout = gr_read_layout(params);
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::T_SCRATCH});
        if (m_pd) {
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GatedResidualInternalBufIdx::WU_F16)});
        } else {
            args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // W_up
        }
        if (m_q) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(GrOp::ReadInputIdx::SCALE_UP)});
            args.push_back({ArgumentDescriptor::Types::INPUT, gr_zp_input(layout, GrOp::ReadInputIdx::ZP_UP, GrOp::ReadInputIdx::SCALE_UP)});
        }
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::RBAR});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // mixed
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // num_rows
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[tier = m_tier, q = m_q, pd = m_pd](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = rows_of(params, *desc);
            const bool share = q && !gr_decode_like(tier);
            const bool bsplit = gr_kperm(params, tier, q, pd);
            const auto tiling = share    ? GrGemmTiling{16, 1, 1}
                                : bsplit ? GrGemmTiling{8, 1, 1}
                                         : gr_up_tiling(*desc, params.get_device_info(), tier, q ? gr_weight_type(params) : GrWType::f16);
            const size_t col_tiles = desc->hidden_size / gr_dpas_n;
            const size_t row_tiles = std::max<size_t>(1, (num_rows + tiling.tm - 1) / tiling.tm);

            auto& wgs = kd.params.workGroups;
            if (bsplit) {
                // one subgroup per branch
                wgs.global = {col_tiles * desc->hc_count * gr_sg, row_tiles, 1};
                wgs.local = {desc->hc_count * gr_sg, 1, 1};
            } else if (share) {
                wgs.global = {col_tiles * gr_up_q_st * gr_sg, (row_tiles + gr_up_q_st - 1) / gr_up_q_st, 1};
                wgs.local = {gr_up_q_st * gr_sg, 1, 1};
            } else if (tiling.ksplit > 1) {
                wgs.global = {col_tiles * tiling.ksplit * gr_sg, row_tiles, 1};
                wgs.local = {tiling.ksplit * gr_sg, 1, 1};
            } else {
                // Work-groups of lws_y token tiles reuse the same W_up columns from L1.
                const size_t lws_y = std::min(tiling.lws_y, row_tiles);
                wgs.global = {col_tiles * gr_sg, (row_tiles + lws_y - 1) / lws_y * lws_y, 1};
                wgs.local = {gr_sg, lws_y, 1};
            }
            push_num_rows(kd, num_rows);
        }};
    }

private:
    GrTier m_tier;
    bool m_q;
    bool m_pd;
};

// ---------------------------------------------------------------------------
// Weight dequantization into an f16 scratch matrix (see gr_pd_min_rows)
// ---------------------------------------------------------------------------
class GrDequantGenerator : public GrGenerator {
public:
    explicit GrDequantGenerator(bool down) : GrGenerator("gr_common", down ? "down" : "up", /*large_grf=*/false), m_down(down) {}

protected:
    static size_t rows_n(const gated_residual& d, bool down) {
        return down ? d.lowrank : d.hc_count * d.hidden_size;
    }
    static size_t cols_k(const gated_residual& d, bool down) {
        return down ? d.hc_count * d.hidden_size : d.lowrank;
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GrGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        const auto wtype = gr_weight_type(params);
        const size_t k = cols_k(*desc, m_down);
        jit.make("GR_DEQUANT_WEIGHTS", 1);
        add_wq_format_constants(jit, params);
        jit.make("GR_DQ_N", rows_n(*desc, m_down));
        jit.make("GR_DQ_K", k);
        add_wq_matrix_constants(jit,
                                "GR_DQ",
                                wtype,
                                gr_wq_kch(wtype, k, k, 64),
                                k,
                                gr_group_size(params, m_down ? GrOp::ReadInputIdx::SCALE_DOWN : GrOp::ReadInputIdx::SCALE_UP, k));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        const auto layout = gr_read_layout(params);
        const uint32_t w_idx = m_down ? GrOp::ReadInputIdx::W_DOWN : GrOp::ReadInputIdx::W_UP;
        const uint32_t scale_idx = m_down ? GrOp::ReadInputIdx::SCALE_DOWN : GrOp::ReadInputIdx::SCALE_UP;
        const uint32_t zp_idx = m_down ? GrOp::ReadInputIdx::ZP_DOWN : GrOp::ReadInputIdx::ZP_UP;
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, w_idx});
        args.push_back({ArgumentDescriptor::Types::INPUT, scale_idx});
        args.push_back({ArgumentDescriptor::Types::INPUT, gr_zp_input(layout, zp_idx, scale_idx)});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER,
                        static_cast<uint32_t>(m_down ? GatedResidualInternalBufIdx::WD_F16 : GatedResidualInternalBufIdx::WU_F16)});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[down = m_down](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto desc = params.typed_desc<gated_residual>();
            const size_t n = rows_n(*desc, down);
            const size_t k = cols_k(*desc, down);
            const size_t kch = gr_wq_kch(gr_weight_type(params), k, k, 64);
            auto& wgs = kd.params.workGroups;
            wgs.global = {n / 16 * gr_sg, k / kch, 1};
            wgs.local = {gr_sg, 1, 1};
        }};
    }

private:
    bool m_down;
};

/// Dequantizes the compressed W_inject into the f16 scratch matrix WI_F16, see gr_common.cl.
class GrDequantInjectGenerator : public GrGenerator {
public:
    GrDequantInjectGenerator() : GrGenerator("gr_common", "inject", /*large_grf=*/false) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GrGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        const size_t k = desc->hc_count * desc->hidden_size;
        jit.make("GR_DEQUANT_INJECT", 1);
        add_wq_format_constants(jit, params);
        jit.make("GR_DQ_N", desc->hc_count);
        jit.make("GR_DQ_K", k);
        jit.make("GR_DQ_GS", gr_group_size(params, gr_read_layout(params).scale_inject, k));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        const auto layout = gr_read_layout(params);
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(layout.w_inject)});
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(layout.scale_inject)});
        args.push_back({ArgumentDescriptor::Types::INPUT, gr_zp_input(layout, layout.zp_inject, layout.scale_inject)});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GatedResidualInternalBufIdx::WI_F16)});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto desc = params.typed_desc<gated_residual>();
            auto& wgs = kd.params.workGroups;
            wgs.global = {desc->hc_count * desc->hc_count * desc->hidden_size, 1, 1};
            wgs.local = {16, 1, 1};
        }};
    }
};

bool use_single_read(const RuntimeParams& params, size_t rows, bool compressed) {
    const auto desc = params.typed_desc<gated_residual>();
    if (!compressed) {
        return use_single_dpas_read(*desc, params.get_device_info(), rows);
    }
    return desc->hidden_size % 32 == 0 && rows * desc->hc_count * desc->hidden_size * 2 > gr_q_scratch_limit(params);
}

/// Dequantize the weights into f16 scratch and run the f16 kernels. A compressed READ with a write gate also takes this route
/// where it would use the compressed single kernel, which has no write-gate tile.
bool use_predequant(const RuntimeParams& params, GrWType t, size_t num_rows) {
    if (t == GrWType::f16) {
        return false;
    }
    return num_rows > gr_pd_min_rows(params) ||
           (params.typed_desc<gated_residual>()->produce_write_gate && use_single_read(params, num_rows, /*compressed=*/true));
}

bool gr_kperm(const RuntimeParams& params, GrTier tier, bool q, bool pd) {
    if (!q || pd || !gr_decode_like(tier) || gr_weight_type(params) != GrWType::u4) {
        return false;
    }
    // the decode RBAR permutes in registers: 128-half chunks, 8 halves per lane
    return gr_best_tile(params.typed_desc<gated_residual>()->hidden_size) == 128 && !use_predequant(params, GrWType::u4, gr_decode_rows) &&
           !use_single_read(params, gr_decode_rows, /*compressed=*/true);
}

class GatedResidualImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GatedResidualImpl)

    Stage::Ptr gr_read = make_stage<GrReadGenerator>(false);
    Stage::Ptr gr_read_decode = make_stage<GrReadGenerator>(true);
    Stage::Ptr gr_write = make_stage<GrWriteGenerator>();
    Stage::Ptr gr_rbar = make_stage<GrRbarGenerator>();
    Stage::Ptr gr_rbar_decode = make_stage<GrRbarGenerator>(true);
    Stage::Ptr gr_down_proj = make_stage<GrDownProjGenerator>(GrTier::wide);
    Stage::Ptr gr_down_proj_mid = make_stage<GrDownProjGenerator>(GrTier::mid);
    Stage::Ptr gr_down_proj_decode = make_stage<GrDownProjGenerator>(GrTier::decode);
    Stage::Ptr gr_up_proj = make_stage<GrUpProjGenerator>(GrTier::wide);
    Stage::Ptr gr_up_proj_decode = make_stage<GrUpProjGenerator>(GrTier::decode);
    Stage::Ptr gr_read_dpas = make_stage<GrReadDpasGenerator>();
    // compressed weights (same kernel files as f16, selected by GR_W_TYPE)
    Stage::Ptr gr_down_proj_q = make_stage<GrDownProjGenerator>(GrTier::wide, true);
    Stage::Ptr gr_down_proj_q_mid = make_stage<GrDownProjGenerator>(GrTier::mid, true);
    Stage::Ptr gr_down_proj_q_decode = make_stage<GrDownProjGenerator>(GrTier::decode, true);
    Stage::Ptr gr_down_proj_q_gemv = make_stage<GrDownProjGenerator>(GrTier::gemv, true);
    Stage::Ptr gr_up_proj_q = make_stage<GrUpProjGenerator>(GrTier::wide, true);
    Stage::Ptr gr_up_proj_q_decode = make_stage<GrUpProjGenerator>(GrTier::decode, true);
    Stage::Ptr gr_read_dpas_q = make_stage<GrReadDpasGenerator>(true);
    // compressed weights dequantized into f16 scratch by gr_dq_*, then the f16 kernels on the scratch (see gr_pd_min_rows)
    Stage::Ptr gr_dq_down = make_stage<GrDequantGenerator>(true);
    Stage::Ptr gr_dq_up = make_stage<GrDequantGenerator>(false);
    Stage::Ptr gr_down_proj_pd = make_stage<GrDownProjGenerator>(GrTier::wide, false, true);
    Stage::Ptr gr_down_proj_pd_mid = make_stage<GrDownProjGenerator>(GrTier::mid, false, true);
    Stage::Ptr gr_down_proj_pd_decode = make_stage<GrDownProjGenerator>(GrTier::decode, false, true);
    Stage::Ptr gr_up_proj_pd = make_stage<GrUpProjGenerator>(GrTier::wide, false, true);
    Stage::Ptr gr_up_proj_pd_decode = make_stage<GrUpProjGenerator>(GrTier::decode, false, true);
    Stage::Ptr gr_read_dpas_pd = make_stage<GrReadDpasGenerator>(false, true);
    // write gate of a compressed READ above the decode regime: W_inject dequantized into f16 scratch, then the f16 gate-only
    // kernel on the Rbar scratch (the decode DOWN_PROJ has the write-gate tile itself)
    Stage::Ptr gr_dq_inject = make_stage<GrDequantInjectGenerator>();
    Stage::Ptr gr_gate_mid = make_stage<GrGateProjGenerator>(GrTier::mid);
    Stage::Ptr gr_gate_wide = make_stage<GrGateProjGenerator>(GrTier::wide);

    /// The dispatches of one READ / FINAL_MIX on the dpas path for `rows` token rows.
    struct ReadPlan {
        bool q;
        bool pd;          // weights dequantized into f16 scratch first
        bool single;      // one gr_read_dpas dispatch
        GrTier tier;      // DOWN_PROJ tier of the 3-dispatch pipeline
        bool gate_stage;  // separate write-gate dispatch (compressed weights, shared DOWN_PROJ)
        bool dq_inject;   // compressed W_inject dequantized into f16 scratch
    };

    static ReadPlan read_plan(const RuntimeParams& params, size_t rows) {
        const auto desc = params.typed_desc<gated_residual>();
        const auto wtype = gr_weight_type(params);
        ReadPlan p;
        p.q = wtype != GrWType::f16;
        p.pd = use_predequant(params, wtype, rows);
        p.single = use_single_read(params, rows, p.q && !p.pd);
        p.tier = gr_down_tier(rows, params.get_device_info());
        const bool gate_q = p.q && desc->produce_write_gate;
        p.gate_stage = gate_q && !p.pd && !p.single && !gr_decode_like(p.tier);
        p.dq_inject = gate_q && (p.pd || p.gate_stage);
        return p;
    }

    GatedResidualImpl() : PrimitiveImplOCL(GatedResidualImplementationManager::get_type_info_static()) {}
    GatedResidualImpl(const program_node& /*node*/, const RuntimeParams& params) : GatedResidualImpl() {
        const auto desc = params.typed_desc<gated_residual>();
        if (!params.is_dynamic()) {
            for (auto* stage : stages_for(params, rows_of(params, *desc))) {
                add_stage(*stage, params);
            }
            return;
        }
        // A dynamic shape builds every stage some row count can take.
        if (desc->mode == gated_residual::Mode::WRITE) {
            add_stage(gr_write, params);
        } else if (can_use_dpas_gr_read(params)) {
            const bool q = gr_weight_type(params) != GrWType::f16;
            add_stage(gr_rbar, params);
            add_stage(gr_rbar_decode, params);
            for (auto* stage : {&down_stage(GrTier::decode, q),
                                &down_stage(GrTier::mid, q),
                                &down_stage(GrTier::wide, q),
                                &up_stage(GrTier::decode, q),
                                &up_stage(GrTier::wide, q)}) {
                add_stage(*stage, params);
            }
            if (desc->hidden_size % 32 == 0) {
                add_stage(q ? gr_read_dpas_q : gr_read_dpas, params);
            }
            if (q) {
                add_stage(gr_down_proj_q_gemv, params);
                for (auto* stage :
                     {&gr_dq_down, &gr_dq_up, &gr_down_proj_pd_decode, &gr_down_proj_pd_mid, &gr_down_proj_pd, &gr_up_proj_pd_decode, &gr_up_proj_pd}) {
                    add_stage(*stage, params);
                }
                if (desc->hidden_size % 32 == 0) {
                    add_stage(gr_read_dpas_pd, params);
                }
            }
            if (q && desc->produce_write_gate) {
                for (auto* stage : {&gr_dq_inject, &gr_gate_mid, &gr_gate_wide}) {
                    add_stage(*stage, params);
                }
            }
        } else {
            add_stage(gr_read_decode, params);
            add_stage(gr_read, params);
        }
    }

    /// The stages one call over `rows` token rows dispatches, in order: the scratch buffers are produced and consumed in it.
    std::vector<Stage::Ptr*> stages_for(const RuntimeParams& params, size_t rows) {
        const auto desc = params.typed_desc<gated_residual>();
        if (desc->mode == gated_residual::Mode::WRITE) {
            return {&gr_write};
        }
        if (!can_use_dpas_gr_read(params)) {
            return {rows <= gr_decode_rows ? &gr_read_decode : &gr_read};
        }
        const auto p = read_plan(params, rows);
        std::vector<Stage::Ptr*> stages;
        if (p.pd) {
            stages.push_back(&gr_dq_down);
            stages.push_back(&gr_dq_up);
        }
        if (p.dq_inject) {
            stages.push_back(&gr_dq_inject);
        }
        if (p.single) {
            stages.push_back(p.pd ? &gr_read_dpas_pd : (p.q ? &gr_read_dpas_q : &gr_read_dpas));
            return stages;
        }
        stages.push_back(&rbar_stage(rows));
        if (p.gate_stage) {
            stages.push_back(&gate_stage(p.tier));
        }
        stages.push_back(&down_stage(p.tier, p.q, p.pd));
        stages.push_back(&up_stage(gr_up_tier(rows), p.q, p.pd));
        return stages;
    }

    Stage::Ptr& rbar_stage(size_t rows) {
        return rows <= gr_decode_rows ? gr_rbar_decode : gr_rbar;
    }

    Stage::Ptr& down_stage(GrTier tier, bool q = false, bool pd = false) {
        if (pd) {
            return gr_decode_like(tier) ? gr_down_proj_pd_decode : tier == GrTier::mid ? gr_down_proj_pd_mid : gr_down_proj_pd;
        }
        if (q) {
            if (tier == GrTier::gemv) {
                return gr_down_proj_q_gemv;
            }
            return tier == GrTier::decode ? gr_down_proj_q_decode : tier == GrTier::mid ? gr_down_proj_q_mid : gr_down_proj_q;
        }
        return gr_decode_like(tier) ? gr_down_proj_decode : tier == GrTier::mid ? gr_down_proj_mid : gr_down_proj;
    }

    Stage::Ptr& up_stage(GrTier tier, bool q = false, bool pd = false) {
        if (pd) {
            return tier == GrTier::decode ? gr_up_proj_pd_decode : gr_up_proj_pd;
        }
        if (q) {
            return tier == GrTier::decode ? gr_up_proj_q_decode : gr_up_proj_q;
        }
        return tier == GrTier::decode ? gr_up_proj_decode : gr_up_proj;
    }

    Stage::Ptr& gate_stage(GrTier tier) {
        return tier == GrTier::mid ? gr_gate_mid : gr_gate_wide;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatedResidualImpl>(this);
    }

    // A dynamic impl holds every stage; the row count picks the ones to run.
    [[nodiscard]] std::vector<size_t> get_stages_execution_order(const kernel_impl_params& params) const override {
        const size_t rows = rows_of(params, *params.typed_desc<gated_residual>());
        // stages_for only reads the identities of the stage members
        const auto stages = const_cast<GatedResidualImpl*>(this)->stages_for(params, rows);
        std::vector<size_t> order;
        for (const auto* stage : stages) {
            const auto it = std::find(_stages.begin(), _stages.end(), stage->get());
            OPENVINO_ASSERT(it != _stages.end() && has_stage(*stage), "[GPU][OCL] gated_residual: a stage for ", rows, " rows was not built");
            order.push_back(static_cast<size_t>(it - _stages.begin()));
        }
        return order;
    }

    // The scratch buffers exist only for the static dpas READ. Their indices are fixed (the dynamic impl's kernels bind them
    // before the row count is known), so a buffer the plan does not use is a 1-element placeholder.
    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        if (params.is_dynamic()) {
            return {};
        }
        const auto desc = params.typed_desc<gated_residual>();
        if (desc->mode == gated_residual::Mode::WRITE || !can_use_dpas_gr_read(params)) {
            return {};
        }
        const size_t cd = desc->hc_count * desc->hidden_size;
        const size_t num_rows = std::max<size_t>(1, get_num_rows(params.get_input_layout(0), cd));
        const auto p = read_plan(params, num_rows);
        const auto size = [](bool used, size_t count) {
            return used ? count : size_t{1};
        };
        return {BufferDescriptor{size(!p.single, num_rows * cd), ov::element::f16},             // RBAR
                BufferDescriptor{size(!p.single, num_rows * desc->lowrank), ov::element::f16},  // T_SCRATCH
                BufferDescriptor{size(p.pd, desc->lowrank * cd), ov::element::f16},             // WD_F16
                BufferDescriptor{size(p.pd, cd * desc->lowrank), ov::element::f16},             // WU_F16
                BufferDescriptor{size(p.dq_inject, desc->hc_count * cd), ov::element::f16}};    // WI_F16
    }
};

}  // namespace

bool GatedResidualImplementationManager::validate_impl(const cldnn::program_node& node) const {
    assert(node.is_type<cldnn::gated_residual>());

    if (node.has_fused_primitives()) {
        return false;
    }

    const auto& info = node.get_program().get_engine().get_device_info();
    if (!caps::is_device_supported(info)) {
        return false;
    }

    const auto desc = node.as<cldnn::gated_residual>().get_primitive();
    const bool systolic = caps::has_systolic(info, node.get_program().get_config());
    if (!caps::is_geometry_supported(desc->mode, desc->hc_count, desc->hidden_size, desc->lowrank, systolic)) {
        return false;
    }

    // Compressed READ / FINAL_MIX: i4 / u4 / i8 / u8 W_down / W_up (and W_inject) with f16 scales and optional zero points of
    // the weight type, on the dpas kernels and on the scalar gr_read.cl.
    const size_t input_count = node.get_dependencies().size();
    std::optional<GrOp::ReadLayout> layout;
    if (desc->mode != cldnn::gated_residual::Mode::WRITE) {
        layout = GrOp::read_layout(input_count, desc->produce_write_gate);
        if (!layout) {
            return false;
        }
    }
    if (layout && layout->compressed) {
        const auto wt = node.get_input_layout(2).data_type;
        if (!GrOp::is_compressed_weight_type(wt) || node.get_input_layout(3).data_type != wt) {
            return false;
        }
        for (size_t i = 0; i < input_count; i++) {
            const auto& l = node.get_input_layout(i);
            const bool is_scale =
                i < 2 || i == GrOp::ReadInputIdx::SCALE_DOWN || i == GrOp::ReadInputIdx::SCALE_UP || (desc->produce_write_gate && i == layout->scale_inject);
            if (l.format != cldnn::format::bfyx || l.data_type != (is_scale ? ov::element::Type_t::f16 : wt)) {
                return false;
            }
        }
        const auto& sd = node.get_input_layout(GrOp::ReadInputIdx::SCALE_DOWN);
        const auto& su = node.get_input_layout(GrOp::ReadInputIdx::SCALE_UP);
        if (sd.is_dynamic() || su.is_dynamic() || sd.get_partial_shape().size() != 2 || su.get_partial_shape().size() != 2) {
            return false;
        }
        const size_t cd = desc->hc_count * desc->hidden_size;
        const size_t gd = sd.get_shape()[0];
        const size_t gu = su.get_shape()[0];
        const size_t bits = ov::element::Type(wt).bitwidth();
        if (sd.get_shape()[1] != desc->lowrank || su.get_shape()[1] != cd || !caps::is_matrix_compression_supported(bits, desc->lowrank, cd, gd) ||
            !caps::is_matrix_compression_supported(bits, cd, desc->lowrank, gu)) {
            return false;
        }
        if (desc->produce_write_gate) {
            const auto& wi = node.get_input_layout(layout->w_inject);
            const auto& si = node.get_input_layout(layout->scale_inject);
            if (wi.is_dynamic() || si.is_dynamic() || wi.get_partial_shape().size() != 2 || si.get_partial_shape().size() != 2 ||
                wi.get_shape()[0] != desc->hc_count || wi.get_shape()[1] != cd || si.get_shape()[1] != desc->hc_count ||
                !caps::is_row_compression_supported(bits, cd, si.get_shape()[0])) {
                return false;
            }
        }
    } else {
        for (size_t i = 0; i < node.get_dependencies().size(); i++) {
            const auto& l = node.get_input_layout(i);
            if (l.format != cldnn::format::bfyx || l.data_type != ov::element::f16) {
                return false;
            }
        }
    }
    for (size_t i = 0; i < node.get_outputs_count(); i++) {
        const auto& l = node.get_output_layout(i);
        if (l.format != cldnn::format::bfyx || l.data_type != ov::element::f16) {
            return false;
        }
    }
    return true;
}

std::unique_ptr<cldnn::primitive_impl> GatedResidualImplementationManager::create_impl(const cldnn::program_node& node,
                                                                                       const cldnn::kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::gated_residual>());
    return std::make_unique<GatedResidualImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GatedResidualImpl)
