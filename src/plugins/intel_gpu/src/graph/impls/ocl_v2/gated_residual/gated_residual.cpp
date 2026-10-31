// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_residual.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "gated_residual_stage_events.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/gated_residual.hpp"
#include "registry/implementation_manager.hpp"

namespace ov::intel_gpu::ocl {

using namespace cldnn;

namespace {

constexpr size_t gr_sg = 16;           // subgroup size == one Xe2 hardware thread
constexpr size_t gr_wg_size = 16;      // subgroups of the scalar / single-kernel work-groups
constexpr size_t gr_dpas_k_step = 32;  // K consumed per dpas loop step (two K16 slices)
constexpr size_t gr_dpas_n = 16;       // dpas N axis == GRF width on Xe2
constexpr size_t gr_down_tn = 32;      // low-rank columns per down-projection subgroup
constexpr size_t gr_decode_rows = 32;  // up to here both GEMMs use decode tiling
constexpr size_t gr_read_dpas_tm = 32;
constexpr size_t gr_read_dpas_nt = 2;

/// Per-stage GPU events recorded by the most recent GatedResidualImpl::execute() call (thread local: networks may run on
/// different threads); exposed through gr_debug::take_stage_events().
thread_local std::vector<std::pair<std::string, event::ptr>> g_gr_stage_events;

void record_stage_event(const char* name, const event::ptr& ev) {
#ifdef GPU_DEBUG_CONFIG
    if (ev != nullptr) {
        g_gr_stage_events.emplace_back(name, ev);
    }
#else
    (void)name;
    (void)ev;
#endif
}

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

/// Row-count regimes of the two dpas GEMM stages: few rows need the K reduction split over a work-group so the weights are
/// streamed by enough threads; the split stops paying off once the token tiles alone fill the GPU.
enum class GrTier { decode, mid, wide };

GrTier gr_down_tier(size_t num_rows, const cldnn::device_info& info) {
    if (num_rows <= gr_decode_rows) {
        return GrTier::decode;
    }
    return num_rows <= 4 * static_cast<size_t>(info.execution_units_count) ? GrTier::mid : GrTier::wide;
}

GrTier gr_up_tier(size_t num_rows) {
    return num_rows <= gr_decode_rows ? GrTier::decode : GrTier::wide;
}

struct GrGemmTiling {
    size_t tm;      // tokens per subgroup
    size_t ksplit;  // subgroups of a work-group splitting the K reduction
    size_t lws_y;   // token tiles per work-group (ksplit == 1 only)
};

GrGemmTiling gr_down_tiling(const gated_residual& desc, const cldnn::device_info& info, GrTier tier, bool compressed = false) {
    const size_t cd_steps = desc.hc_count * desc.hidden_size / gr_dpas_k_step;
    // Dequantizing makes every K step longer, so compressed weights want more subgroups splitting K.
    const size_t boost = compressed ? 2 : 1;
    switch (tier) {
    case GrTier::decode:
        return {16, largest_divisor_up_to(cd_steps, 8 * boost), 1};
    case GrTier::mid:
        return {32, largest_divisor_up_to(cd_steps, (info.execution_units_count >= 128 ? 4 : 2) * boost), 1};
    default:
        return {32, 1, 1};
    }
}

GrGemmTiling gr_up_tiling(const gated_residual& desc, const cldnn::device_info& info, GrTier tier, bool compressed = false) {
    if (tier != GrTier::decode) {
        return {16, 1, 8};
    }
    // Narrow D leaves few column tiles; split L until there are ~2 subgroups per EU. Compressed weights move fewer bytes per
    // K step and split further, down to two K32 steps per subgroup (PTL decode UP_PROJ: u4 1.3-1.6x, u8 1.1-1.2x faster).
    const size_t col_tiles = desc.hidden_size / gr_dpas_n;
    const size_t eu = info.execution_units_count;
    const size_t cap = compressed ? std::min((8 * eu + col_tiles - 1) / col_tiles, desc.lowrank / (2 * gr_dpas_k_step)) : 2 * eu / col_tiles;
    return {8, largest_divisor_up_to(desc.lowrank / gr_dpas_k_step, std::max<size_t>(1, cap)), 1};
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

bool xmx_available(const cldnn::device_info& info) {
    return info.arch >= gpu_arch::xe2 && info.supports_immad;
}

/// Whether the systolic READ/FINAL_MIX path can replace the fused scalar gr_read kernel.
bool can_use_dpas_gr_read(const gated_residual& desc, const cldnn::device_info& info) {
    if (!xmx_available(info)) {
        return false;
    }
    const size_t cd = desc.hc_count * desc.hidden_size;
    if (cd < 32 || cd % 32 != 0) {
        return false;
    }
    if (desc.lowrank < 32 || desc.lowrank % 32 != 0) {
        return false;
    }
    // the write gate is one N16 column tile stored in half pairs
    return desc.hc_count != 0 && desc.hc_count % 2 == 0 && desc.hc_count <= gr_dpas_n;
}

/// Weight storage of the READ / FINAL_MIX kernels (GR_W_TYPE): f16, or the asymmetric u4 (group 32) / u8 (per-channel) forms
/// that arrive as the 8-input compressed op (W_down, W_up, scale_down, scale_up, zp_down, zp_up).
enum class GrWType : int { f16 = 0, u4 = 1, u8 = 2 };

GrWType gr_weight_type(const RuntimeParams& params) {
    const auto dt = params.get_input_layout(2).data_type;
    return dt == ov::element::u4 ? GrWType::u4 : (dt == ov::element::u8 ? GrWType::u8 : GrWType::f16);
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

constexpr size_t gr_wq_down_load_bytes = 32;  // DOWN: per-lane weight bytes per step (32 measured faster than 64 on PTL)
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
size_t gr_q_scratch_limit() {
    static const size_t bytes = [] {
        const char* env = std::getenv("OV_GPU_GR_Q_SCRATCH_LIMIT_MB");  // lets tests reach the single-kernel path with few rows
        return (env != nullptr ? static_cast<size_t>(std::strtoull(env, nullptr, 10)) : size_t{256}) << 20;
    }();
    return bytes;
}

bool use_single_read(const gated_residual& desc, const cldnn::device_info& info, size_t rows, bool compressed);

/// Compressed READ above this many rows dequantizes both matrices into f16 scratch buffers with a separate kernel and then runs
/// the f16 kernels; below it the compute kernels dequantize on the fly. OV_GPU_GR_PD_MIN_ROWS overrides the threshold.
size_t gr_pd_min_rows(const cldnn::device_info& info) {
    static const long long env = [] {
        const char* e = std::getenv("OV_GPU_GR_PD_MIN_ROWS");
        return e != nullptr ? std::strtoll(e, nullptr, 10) : -1LL;
    }();
    // Measured on PTL (B390, 96 EU): the dequantizing GEMM kernels win up to ~4 rows per EU, the separate dequantization plus
    // the f16 single kernel (no Rbar / T scratch traffic, weights hot in cache) wins above ~5 rows per EU.
    return env >= 0 ? static_cast<size_t>(env) : 5 * static_cast<size_t>(info.execution_units_count);
}

/// GR_<prefix>_KCH / _NRAW / _RB of one weight matrix [N, K].
void add_wq_matrix_constants(JitConstants& jit, const std::string& prefix, GrWType t, size_t kch, size_t row_k) {
    jit.make(prefix + "_KCH", kch);
    jit.make(prefix + "_NRAW", kch * gr_weight_bits(t) / 32);
    jit.make(prefix + "_RB", row_k * gr_weight_bits(t) / 8);
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
        if (m_large_grf && params.get_device_info().arch >= gpu_arch::xe_hpg) {
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

    [[nodiscard]] static Arguments read_arguments(const gated_residual& desc, GrWType wtype, bool pd = false) {
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
            args.push_back({ArgumentDescriptor::Types::INPUT, 4});  // scale_down [G, L]
            args.push_back({ArgumentDescriptor::Types::INPUT, 5});  // scale_up [G, C*D]
            args.push_back({ArgumentDescriptor::Types::INPUT, 6});  // zp_down
            args.push_back({ArgumentDescriptor::Types::INPUT, 7});  // zp_up
        }
        if (desc.produce_write_gate) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 4});  // W_inject
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // mixed
        if (desc.produce_write_gate) {
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
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return read_arguments(*params.typed_desc<gated_residual>(), GrWType::f16);
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
        jit.make("GR_PRODUCE_WRITE_GATE", desc->produce_write_gate ? 1 : 0);
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
            jit.make("GR_W_TYPE", static_cast<int>(wtype));
            add_wq_matrix_constants(jit, "GR_WD", wtype, gr_wq_kch(wtype, desc->hidden_size, cd, gr_wq_dpas_load_bytes), cd);
            add_wq_matrix_constants(jit, "GR_WU", wtype, gr_wq_kch(wtype, desc->lowrank, desc->lowrank, gr_wq_dpas_load_bytes), desc->lowrank);
        }
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return read_arguments(*params.typed_desc<gated_residual>(), m_q ? gr_weight_type(params) : GrWType::f16, m_pd);
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
    GrRbarGenerator() : GrGenerator("gr_read_pipeline", "rbar") {}

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
        jit.make("GR_NCH", nch);
        // Gain chunks are independent of the norm: preload up to 16 (64 GRFs) while the row is still being reduced.
        jit.make("GR_GPF", desc->hidden_size <= 2560 ? std::min<size_t>(nch, 16) : 0);
        return jit;
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
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = rows_of(params, *desc);
            auto& wgs = kd.params.workGroups;
            // One subgroup per (branch, row): the norm is branch-wise, so no reduction across subgroups.
            wgs.global = {desc->hc_count * gr_sg, std::max<size_t>(1, num_rows), 1};
            wgs.local = {desc->hc_count * gr_sg, 1, 1};
            push_num_rows(kd, num_rows);
        }};
    }
};

class GrDownProjGenerator : public GrGenerator {
public:
    explicit GrDownProjGenerator(GrTier tier, bool q = false, bool pd = false)
        : GrGenerator("gr_read_pipeline", gr_tier_suffix(tier, true, q, pd)),
          m_tier(tier),
          m_q(q),
          m_pd(pd) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GrGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        const auto wtype = m_q ? gr_weight_type(params) : GrWType::f16;
        const bool share = m_q && m_tier != GrTier::decode;
        const auto qs = gr_down_q_share(*desc, m_tier);
        const auto tiling = share ? GrGemmTiling{32, qs.ks, 1} : gr_down_tiling(*desc, params.get_device_info(), m_tier, wtype != GrWType::f16);
        jit.make("GR_PIPELINE_STAGE", 1);
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.make("GR_DOWN_TM", tiling.tm);
        jit.make("GR_DOWN_TN", gr_down_tn);
        jit.make("GR_DOWN_KSPLIT", tiling.ksplit);
        jit.make("GR_PRODUCE_WRITE_GATE", desc->produce_write_gate ? 1 : 0);
        if (m_q) {
            const size_t cd = desc->hc_count * desc->hidden_size;
            jit.make("GR_W_TYPE", static_cast<int>(wtype));
            jit.make("GR_Q_SHARE", share ? 1 : 0);
            if (share) {
                jit.make("GR_Q_ST", qs.st);
            }
            add_wq_matrix_constants(jit, "GR_WD", wtype, share ? 32 : gr_wq_kch(wtype, cd / tiling.ksplit, cd, gr_wq_down_load_bytes), cd);
        }
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        const auto desc = params.typed_desc<gated_residual>();
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::RBAR});
        if (m_pd) {
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GatedResidualInternalBufIdx::WD_F16)});
        } else {
            args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // W_down
        }
        if (m_q) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 4});  // scale_down
            args.push_back({ArgumentDescriptor::Types::INPUT, 6});  // zp_down
        }
        if (desc->produce_write_gate) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 4});  // W_inject
        }
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::T_SCRATCH});
        if (desc->produce_write_gate) {
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
            const bool share = q && tier != GrTier::decode;
            const auto qs = gr_down_q_share(*desc, tier);
            const auto tiling =
                share ? GrGemmTiling{32, qs.ks, 1} : gr_down_tiling(*desc, params.get_device_info(), tier, q && gr_weight_type(params) != GrWType::f16);
            const size_t col_tiles = desc->lowrank / gr_down_tn + (desc->produce_write_gate ? 1 : 0);
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
        const bool share = m_q && m_tier != GrTier::decode;
        const auto tiling = share ? GrGemmTiling{16, 1, 1} : gr_up_tiling(*desc, params.get_device_info(), m_tier, m_q);
        jit.make("GR_PIPELINE_STAGE", 2);
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.make("GR_UP_TM", tiling.tm);
        jit.make("GR_UP_KSPLIT", tiling.ksplit);
        if (m_q) {
            const auto wtype = gr_weight_type(params);
            jit.make("GR_W_TYPE", static_cast<int>(wtype));
            jit.make("GR_Q_SHARE", share ? 1 : 0);
            if (share) {
                jit.make("GR_Q_ST", gr_up_q_st);
            }
            add_wq_matrix_constants(jit,
                                    "GR_WU",
                                    wtype,
                                    share ? 32 : gr_wq_kch(wtype, desc->lowrank / tiling.ksplit, desc->lowrank, gr_wq_up_load_bytes),
                                    desc->lowrank);
        }
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::T_SCRATCH});
        if (m_pd) {
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(GatedResidualInternalBufIdx::WU_F16)});
        } else {
            args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // W_up
        }
        if (m_q) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 5});  // scale_up
            args.push_back({ArgumentDescriptor::Types::INPUT, 7});  // zp_up
        }
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::RBAR});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // mixed
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
            const bool share = q && tier != GrTier::decode;
            const auto tiling = share ? GrGemmTiling{16, 1, 1} : gr_up_tiling(*desc, params.get_device_info(), tier, q);
            const size_t col_tiles = desc->hidden_size / gr_dpas_n;
            const size_t row_tiles = std::max<size_t>(1, (num_rows + tiling.tm - 1) / tiling.tm);

            auto& wgs = kd.params.workGroups;
            if (share) {
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
// Weight dequantization into an f16 scratch matrix (experiment, see gr_pd_min_rows)
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
        jit.make("GR_W_TYPE", static_cast<int>(wtype));
        jit.make("GR_DQ_N", rows_n(*desc, m_down));
        jit.make("GR_DQ_K", k);
        add_wq_matrix_constants(jit, "GR_DQ", wtype, gr_wq_kch(wtype, k, k, 64), k);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, m_down ? 2u : 3u});  // weights
        args.push_back({ArgumentDescriptor::Types::INPUT, m_down ? 4u : 5u});  // scale
        args.push_back({ArgumentDescriptor::Types::INPUT, m_down ? 6u : 7u});  // zp
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

bool use_predequant(GrWType t, size_t num_rows, const cldnn::device_info& info) {
    return t != GrWType::f16 && num_rows > gr_pd_min_rows(info);
}

bool use_single_read(const gated_residual& desc, const cldnn::device_info& info, size_t rows, bool compressed) {
    if (!compressed) {
        return use_single_dpas_read(desc, info, rows);
    }
    return desc.hidden_size % 32 == 0 && rows * desc.hc_count * desc.hidden_size * 2 > gr_q_scratch_limit();
}

class GatedResidualImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GatedResidualImpl)

    Stage::Ptr gr_read = make_stage<GrReadGenerator>(false);
    Stage::Ptr gr_read_decode = make_stage<GrReadGenerator>(true);
    Stage::Ptr gr_write = make_stage<GrWriteGenerator>();
    Stage::Ptr gr_rbar = make_stage<GrRbarGenerator>();
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
    Stage::Ptr gr_up_proj_q = make_stage<GrUpProjGenerator>(GrTier::wide, true);
    Stage::Ptr gr_up_proj_q_decode = make_stage<GrUpProjGenerator>(GrTier::decode, true);
    Stage::Ptr gr_read_dpas_q = make_stage<GrReadDpasGenerator>(true);
    // compressed weights dequantized into f16 scratch by gr_dq_*, then the f16 kernels on the scratch (OV_GPU_GR_PD_MIN_ROWS)
    Stage::Ptr gr_dq_down = make_stage<GrDequantGenerator>(true);
    Stage::Ptr gr_dq_up = make_stage<GrDequantGenerator>(false);
    Stage::Ptr gr_down_proj_pd = make_stage<GrDownProjGenerator>(GrTier::wide, false, true);
    Stage::Ptr gr_down_proj_pd_mid = make_stage<GrDownProjGenerator>(GrTier::mid, false, true);
    Stage::Ptr gr_down_proj_pd_decode = make_stage<GrDownProjGenerator>(GrTier::decode, false, true);
    Stage::Ptr gr_up_proj_pd = make_stage<GrUpProjGenerator>(GrTier::wide, false, true);
    Stage::Ptr gr_up_proj_pd_decode = make_stage<GrUpProjGenerator>(GrTier::decode, false, true);
    Stage::Ptr gr_read_dpas_pd = make_stage<GrReadDpasGenerator>(false, true);

    GatedResidualImpl() : PrimitiveImplOCL(GatedResidualImplementationManager::get_type_info_static()) {}
    GatedResidualImpl(const program_node& /*node*/, const RuntimeParams& params) : GatedResidualImpl() {
        const auto desc = params.typed_desc<gated_residual>();
        const auto& info = params.get_device_info();
        // READ and WRITE cannot be merged: an attention / GDN / MLP sub-layer sits between them.
        if (desc->mode == gated_residual::Mode::WRITE) {
            add_stage(gr_write, params);
        } else if (can_use_dpas_gr_read(*desc, info)) {
            // Producer/consumer order through the RBAR/T scratch buffers is enforced by execute(); static shapes only build
            // what their row count needs.
            const auto wtype = gr_weight_type(params);
            const bool q = wtype != GrWType::f16;
            if (params.is_dynamic()) {
                add_stage(gr_rbar, params);
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
                    for (auto* stage :
                         {&gr_dq_down, &gr_dq_up, &gr_down_proj_pd_decode, &gr_down_proj_pd_mid, &gr_down_proj_pd, &gr_up_proj_pd_decode, &gr_up_proj_pd}) {
                        add_stage(*stage, params);
                    }
                    if (desc->hidden_size % 32 == 0) {
                        add_stage(gr_read_dpas_pd, params);
                    }
                }
            } else {
                const size_t rows = rows_of(params, *desc);
                const bool pd = use_predequant(wtype, rows, info);
                if (pd) {
                    add_stage(gr_dq_down, params);
                    add_stage(gr_dq_up, params);
                }
                if (use_single_read(*desc, info, rows, q && !pd)) {
                    add_stage(pd ? gr_read_dpas_pd : (q ? gr_read_dpas_q : gr_read_dpas), params);
                } else {
                    add_stage(gr_rbar, params);
                    add_stage(down_stage(gr_down_tier(rows, info), q, pd), params);
                    add_stage(up_stage(gr_up_tier(rows), q, pd), params);
                }
            }
        } else if (params.is_dynamic()) {
            add_stage(gr_read_decode, params);
            add_stage(gr_read, params);
        } else {
            add_stage(rows_of(params, *desc) <= gr_decode_rows ? gr_read_decode : gr_read, params);
        }
    }

    Stage::Ptr& down_stage(GrTier tier, bool q = false, bool pd = false) {
        if (pd) {
            return tier == GrTier::decode ? gr_down_proj_pd_decode : tier == GrTier::mid ? gr_down_proj_pd_mid : gr_down_proj_pd;
        }
        if (q) {
            return tier == GrTier::decode ? gr_down_proj_q_decode : tier == GrTier::mid ? gr_down_proj_q_mid : gr_down_proj_q;
        }
        return tier == GrTier::decode ? gr_down_proj_decode : tier == GrTier::mid ? gr_down_proj_mid : gr_down_proj;
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

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatedResidualImpl>(this);
    }

    // The default execute() would only expose the LAST dispatched kernel's event, which is wrong for the 3-dispatch READ
    // pipeline; this override chains the stages explicitly and records every stage's own event for the roofline tests.
    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        kernel_dump_info.clear_entries();
        if (instance.can_be_optimized()) {
            return instance.get_network().get_stream().aggregate_events(events, events.size() > 1, instance.is_output());
        }
        update_rt_params(instance);
        g_gr_stage_events.clear();

        if (has_stage(gr_write)) {
            auto ev = execute_stage(events, instance, gr_write);
            record_stage_event("WRITE", ev);
            return ev;
        }
        const auto& params = *instance.get_impl_params();
        const auto desc = params.typed_desc<gated_residual>();
        const size_t rows = rows_of(params, *desc);
        if (has_stage(gr_rbar) || has_stage(gr_read_dpas) || has_stage(gr_read_dpas_q) || has_stage(gr_read_dpas_pd)) {
            const auto wtype = gr_weight_type(params);
            const bool q = wtype != GrWType::f16;
            const bool pd = use_predequant(wtype, rows, params.get_device_info());
            std::vector<event::ptr> deps = events;
            std::vector<event::ptr> all;
            if (pd) {
                OPENVINO_ASSERT(has_stage(gr_dq_down) && has_stage(gr_dq_up), "[GPU][OCL] gated_residual: dequantize stage for ", rows, " rows was not built");
                auto ev_dq_down = execute_stage(events, instance, gr_dq_down);
                record_stage_event("DEQUANT", ev_dq_down);
                auto ev_dq_up = execute_stage({ev_dq_down}, instance, gr_dq_up);
                record_stage_event("DEQUANT", ev_dq_up);
                deps = {ev_dq_up};
                all = {ev_dq_down, ev_dq_up};
            }
            auto& single = pd ? gr_read_dpas_pd : (q ? gr_read_dpas_q : gr_read_dpas);
            if (has_stage(single) && use_single_read(*desc, params.get_device_info(), rows, q && !pd)) {
                auto ev = execute_stage(deps, instance, single);
                record_stage_event("READ_DPAS", ev);
                if (!pd) {
                    return ev;
                }
                all.push_back(ev);
                return instance.get_network().get_stream().aggregate_events(all, true, instance.is_output());
            }
            auto& down = down_stage(gr_down_tier(rows, params.get_device_info()), q, pd);
            auto& up = up_stage(gr_up_tier(rows), q, pd);
            OPENVINO_ASSERT(has_stage(gr_rbar) && has_stage(down) && has_stage(up), "[GPU][OCL] gated_residual: dpas stage for ", rows, " rows was not built");

            auto ev_rbar = execute_stage(deps, instance, gr_rbar);
            record_stage_event("RBAR", ev_rbar);
            auto ev_down = execute_stage({ev_rbar}, instance, down);
            record_stage_event("DOWN_PROJ", ev_down);
            auto ev_up = execute_stage({ev_down}, instance, up);
            record_stage_event("UP_PROJ", ev_up);
            all.insert(all.end(), {ev_rbar, ev_down, ev_up});
            return instance.get_network().get_stream().aggregate_events(all, true, instance.is_output());
        }
        auto& read = rows <= gr_decode_rows ? gr_read_decode : gr_read;
        OPENVINO_ASSERT(has_stage(read), "[GPU][OCL] gated_residual: read stage for ", rows, " rows was not built");
        auto ev = execute_stage(events, instance, read);
        record_stage_event("READ", ev);
        return ev;
    }

    // RBAR / T scratch exist only for the static 3-dispatch pipeline.
    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        if (params.is_dynamic()) {
            return {};
        }
        const auto desc = params.typed_desc<gated_residual>();
        if (desc->mode == gated_residual::Mode::WRITE || !can_use_dpas_gr_read(*desc, params.get_device_info())) {
            return {};
        }
        const size_t cd = desc->hc_count * desc->hidden_size;
        const size_t num_rows = std::max<size_t>(1, get_num_rows(params.get_input_layout(0), cd));
        const auto wtype = gr_weight_type(params);
        const bool pd = use_predequant(wtype, num_rows, params.get_device_info());
        const bool single = use_single_read(*desc, params.get_device_info(), num_rows, wtype != GrWType::f16 && !pd);
        if (single && !pd) {
            return {};  // gr_read_dpas keeps its intermediates in SLM
        }
        std::vector<BufferDescriptor> bufs;
        // the single kernel does not use RBAR / T_SCRATCH; they only keep the indices of the weight scratch buffers stable
        bufs.emplace_back(single ? 1 : num_rows * cd, ov::element::f16);                                  // RBAR
        bufs.emplace_back(single ? 1 : num_rows * std::max<size_t>(1, desc->lowrank), ov::element::f16);  // T_SCRATCH
        if (pd) {
            bufs.emplace_back(desc->lowrank * cd, ov::element::f16);  // WD_F16
            bufs.emplace_back(cd * desc->lowrank, ov::element::f16);  // WU_F16
        }
        return bufs;
    }
};

}  // namespace

namespace gr_debug {

std::vector<std::pair<std::string, event::ptr>> take_stage_events() {
    std::vector<std::pair<std::string, event::ptr>> result;
    result.swap(g_gr_stage_events);
    return result;
}

}  // namespace gr_debug

bool GatedResidualImplementationManager::validate_impl(const cldnn::program_node& node) const {
    assert(node.is_type<cldnn::gated_residual>());

    if (node.has_fused_primitives()) {
        return false;
    }

    const auto& info = node.get_program().get_engine().get_device_info();
    if (!info.supports_intel_subgroups || !info.supports_intel_subgroups_short) {
        return false;
    }

    const auto desc = node.as<cldnn::gated_residual>().get_primitive();
    if (!ov::intel_gpu::op::GatedResidual::is_geometry_supported(desc->mode, desc->hc_count, desc->hidden_size, desc->lowrank)) {
        return false;
    }

    // Compressed READ / FINAL_MIX: u4 (group 32) / u8 (per-channel) W_down / W_up + f16 scale + u8 zp, dpas path only.
    const bool compressed =
        desc->mode != cldnn::gated_residual::Mode::WRITE && node.get_dependencies().size() == ov::intel_gpu::op::GatedResidual::compressed_input_count;
    if (compressed) {
        if (!ov::intel_gpu::op::GatedResidual::is_compression_arch_supported(info) || !can_use_dpas_gr_read(*desc, info)) {
            return false;
        }
        const auto wt = node.get_input_layout(2).data_type;
        if ((wt != ov::element::u4 && wt != ov::element::u8) || node.get_input_layout(3).data_type != wt) {
            return false;
        }
        for (size_t i = 0; i < node.get_dependencies().size(); i++) {
            const auto& l = node.get_input_layout(i);
            const ov::element::Type expected =
                i < 2 ? ov::element::Type(ov::element::f16)
                      : (i < 4 ? ov::element::Type(wt) : (i < 6 ? ov::element::Type(ov::element::f16) : ov::element::Type(ov::element::u8)));
            if (l.format != cldnn::format::bfyx || l.data_type != expected) {
                return false;
            }
        }
        const auto& sd = node.get_input_layout(4);
        const auto& su = node.get_input_layout(5);
        if (sd.is_dynamic() || su.is_dynamic() || sd.get_partial_shape().size() != 2 || su.get_partial_shape().size() != 2) {
            return false;
        }
        const size_t cd = desc->hc_count * desc->hidden_size;
        const size_t gd = sd.get_shape()[0];
        const size_t gu = su.get_shape()[0];
        if (sd.get_shape()[1] != desc->lowrank || su.get_shape()[1] != cd ||
            !ov::intel_gpu::op::GatedResidual::is_weight_compression_supported(gr_weight_bits(wt == ov::element::u4 ? GrWType::u4 : GrWType::u8),
                                                                               desc->lowrank,
                                                                               cd,
                                                                               gd,
                                                                               cd,
                                                                               desc->lowrank,
                                                                               gu,
                                                                               desc->produce_write_gate)) {
            return false;
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
