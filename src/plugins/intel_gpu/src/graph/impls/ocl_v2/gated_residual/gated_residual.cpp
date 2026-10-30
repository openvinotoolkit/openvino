// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_residual.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
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

/// OpenCL is the default; OV_GPU_GR_IMPL=cm hands the primitive to the other implementation.
bool ocl_impl_allowed() {
    const char* v = std::getenv("OV_GPU_GR_IMPL");
    if (v == nullptr) {
        return true;
    }
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return s != "cm";
}

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

GrGemmTiling gr_down_tiling(const gated_residual& desc, const cldnn::device_info& info, GrTier tier) {
    const size_t cd_steps = desc.hc_count * desc.hidden_size / gr_dpas_k_step;
    switch (tier) {
    case GrTier::decode:
        return {16, largest_divisor_up_to(cd_steps, 8), 1};
    case GrTier::mid:
        return {32, largest_divisor_up_to(cd_steps, info.execution_units_count >= 128 ? 4 : 2), 1};
    default:
        return {32, 1, 1};
    }
}

GrGemmTiling gr_up_tiling(const gated_residual& desc, const cldnn::device_info& info, GrTier tier) {
    if (tier != GrTier::decode) {
        return {16, 1, 8};
    }
    // Narrow D leaves few column tiles; split L until there are ~2 subgroups per EU.
    const size_t col_tiles = desc.hidden_size / gr_dpas_n;
    const size_t cap = std::max<size_t>(1, 2 * static_cast<size_t>(info.execution_units_count) / col_tiles);
    return {8, largest_divisor_up_to(desc.lowrank / gr_dpas_k_step, cap), 1};
}

const char* gr_tier_suffix(GrTier tier, bool down) {
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

    [[nodiscard]] static Arguments read_arguments(const gated_residual& desc) {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // residual
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // norm weight
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // W_down
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // W_up
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
        return read_arguments(*params.typed_desc<gated_residual>());
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
    GrReadDpasGenerator() : GrGenerator("gr_read_dpas", "") {}

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
        jit.make("GR_RD_SLM_GAIN", slm_gain);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return read_arguments(*params.typed_desc<gated_residual>());
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
    explicit GrDownProjGenerator(GrTier tier) : GrGenerator("gr_read_pipeline", gr_tier_suffix(tier, true)), m_tier(tier) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GrGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        const auto tiling = gr_down_tiling(*desc, params.get_device_info(), m_tier);
        jit.make("GR_PIPELINE_STAGE", 1);
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.make("GR_DOWN_TM", tiling.tm);
        jit.make("GR_DOWN_TN", gr_down_tn);
        jit.make("GR_DOWN_KSPLIT", tiling.ksplit);
        jit.make("GR_PRODUCE_WRITE_GATE", desc->produce_write_gate ? 1 : 0);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        const auto desc = params.typed_desc<gated_residual>();
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::RBAR});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // W_down
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
        return DispatchDataFunc{[tier = m_tier](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = rows_of(params, *desc);
            const auto tiling = gr_down_tiling(*desc, params.get_device_info(), tier);
            const size_t col_tiles = desc->lowrank / gr_down_tn + (desc->produce_write_gate ? 1 : 0);
            const size_t row_tiles = std::max<size_t>(1, (num_rows + tiling.tm - 1) / tiling.tm);

            auto& wgs = kd.params.workGroups;
            // One work-group per (column tile, token tile); KSPLIT subgroups split the K reduction.
            wgs.global = {col_tiles * tiling.ksplit * gr_sg, row_tiles, 1};
            wgs.local = {tiling.ksplit * gr_sg, 1, 1};
            push_num_rows(kd, num_rows);
        }};
    }

private:
    GrTier m_tier;
};

class GrUpProjGenerator : public GrGenerator {
public:
    explicit GrUpProjGenerator(GrTier tier) : GrGenerator("gr_read_pipeline", gr_tier_suffix(tier, false)), m_tier(tier) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GrGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        const auto tiling = gr_up_tiling(*desc, params.get_device_info(), m_tier);
        jit.make("GR_PIPELINE_STAGE", 2);
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.make("GR_UP_TM", tiling.tm);
        jit.make("GR_UP_KSPLIT", tiling.ksplit);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::T_SCRATCH});
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // W_up
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::RBAR});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // mixed
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
            const auto tiling = gr_up_tiling(*desc, params.get_device_info(), tier);
            const size_t col_tiles = desc->hidden_size / gr_dpas_n;
            const size_t row_tiles = std::max<size_t>(1, (num_rows + tiling.tm - 1) / tiling.tm);

            auto& wgs = kd.params.workGroups;
            if (tiling.ksplit > 1) {
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
};

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
            if (params.is_dynamic()) {
                for (auto* stage : {&gr_rbar, &gr_down_proj_decode, &gr_down_proj_mid, &gr_down_proj, &gr_up_proj_decode, &gr_up_proj}) {
                    add_stage(*stage, params);
                }
                if (desc->hidden_size % 32 == 0) {
                    add_stage(gr_read_dpas, params);
                }
            } else {
                const size_t rows = rows_of(params, *desc);
                if (use_single_dpas_read(*desc, info, rows)) {
                    add_stage(gr_read_dpas, params);
                } else {
                    add_stage(gr_rbar, params);
                    add_stage(down_stage(gr_down_tier(rows, info)), params);
                    add_stage(up_stage(gr_up_tier(rows)), params);
                }
            }
        } else if (params.is_dynamic()) {
            add_stage(gr_read_decode, params);
            add_stage(gr_read, params);
        } else {
            add_stage(rows_of(params, *desc) <= gr_decode_rows ? gr_read_decode : gr_read, params);
        }
    }

    Stage::Ptr& down_stage(GrTier tier) {
        return tier == GrTier::decode ? gr_down_proj_decode : tier == GrTier::mid ? gr_down_proj_mid : gr_down_proj;
    }

    Stage::Ptr& up_stage(GrTier tier) {
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
        if (has_stage(gr_rbar) || has_stage(gr_read_dpas)) {
            if (has_stage(gr_read_dpas) && use_single_dpas_read(*desc, params.get_device_info(), rows)) {
                auto ev = execute_stage(events, instance, gr_read_dpas);
                record_stage_event("READ_DPAS", ev);
                return ev;
            }
            auto& down = down_stage(gr_down_tier(rows, params.get_device_info()));
            auto& up = up_stage(gr_up_tier(rows));
            OPENVINO_ASSERT(has_stage(gr_rbar) && has_stage(down) && has_stage(up), "[GPU][OCL] gated_residual: dpas stage for ", rows, " rows was not built");

            auto ev_rbar = execute_stage(events, instance, gr_rbar);
            record_stage_event("RBAR", ev_rbar);
            auto ev_down = execute_stage({ev_rbar}, instance, down);
            record_stage_event("DOWN_PROJ", ev_down);
            auto ev_up = execute_stage({ev_down}, instance, up);
            record_stage_event("UP_PROJ", ev_up);
            return instance.get_network().get_stream().aggregate_events({ev_rbar, ev_down, ev_up}, true, instance.is_output());
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
        if (use_single_dpas_read(*desc, params.get_device_info(), num_rows)) {
            return {};  // gr_read_dpas keeps its intermediates in SLM
        }
        std::vector<BufferDescriptor> bufs;
        bufs.emplace_back(num_rows * cd, ov::element::f16);                                  // RBAR
        bufs.emplace_back(num_rows * std::max<size_t>(1, desc->lowrank), ov::element::f16);  // T_SCRATCH
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

    if (node.has_fused_primitives() || !ocl_impl_allowed()) {
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

    for (size_t i = 0; i < node.get_dependencies().size(); i++) {
        const auto& l = node.get_input_layout(i);
        if (l.format != cldnn::format::bfyx || l.data_type != ov::element::f16) {
            return false;
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
