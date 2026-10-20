// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_residual.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../primitive_cm_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "gr_stage_events.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/gated_residual.hpp"
#include "registry/implementation_manager.hpp"

namespace ov::intel_gpu::cm {

using namespace cldnn;

namespace {

constexpr size_t gr_wg_size = 16;
constexpr size_t gr_max_lowrank = 512;        // GR_LOWRANK is kept in registers by gr_read.cm
constexpr size_t gr_max_total_width = 16384;  // C*D must fit into SLM as f16
constexpr size_t gr_dpas_k_step = 32;         // K consumed per dpas loop step (two K16 slices)
constexpr size_t gr_dpas_n = 16;              // dpas N axis == GRF width on Xe2
constexpr size_t gr_down_tn = 32;             // low-rank columns per down-projection thread
constexpr size_t gr_decode_rows = 32;         // up to here both GEMMs use decode tiling

constexpr auto get_gr_build_options() {
    return " -cmc -Qxcm_register_file_size=256";
}

/// Per-CM-kernel GPU events recorded by the most recent GatedResidualCMImpl::execute()
/// call, cleared at the start of every call. thread_local because network execution can
/// happen from different threads across tests/streams; TU-local (anonymous namespace)
/// storage, exposed to other translation units only via the accessor function defined at
/// the end of this file (see gr_stage_events.hpp - mirrors qsa.cpp's identical need).
thread_local std::vector<std::pair<std::string, event::ptr>> g_gr_stage_events;

void record_stage_event(const char* name, const event::ptr& ev) {
    if (ev != nullptr) {
        g_gr_stage_events.emplace_back(name, ev);
    }
}

size_t get_num_rows(const layout& l, size_t last_dim) {
    const auto count = l.count();
    OPENVINO_ASSERT(last_dim > 0 && count % last_dim == 0, "[GPU][CM] gated_residual: input element count is not divisible by the last dimension");
    return count / last_dim;
}

JitConstants make_common_jit(const std::shared_ptr<const gated_residual>& desc) {
    JitConstants jit;
    jit.make("GR_BRANCHES", desc->hc_count);
    jit.make("GR_HIDDEN", desc->hidden_size);
    return jit;
}

/// Widest elements-per-thread tile (a multiple of 16) that evenly divides `width`. Used by
/// gr_write.cm (branch loop moved inside the kernel); 16 is always a valid fallback since
/// GR_HIDDEN % 16 == 0 is enforced by validate_impl.
size_t gr_best_tile(size_t width) {
    for (size_t t : {size_t{128}, size_t{64}, size_t{32}, size_t{16}}) {
        if (width % t == 0) {
            return t;
        }
    }
    return 16;
}

/// gr_read.cm work-group size: Xe2+ runs 32 threads per work-group in 256-GRF mode, which
/// halves the per-row latency of the two GEMVs; older devices keep the proven 16.
size_t gr_read_wg_size(const cldnn::device_info& info) {
    return info.arch >= gpu_arch::xe2 ? 32 : gr_wg_size;
}

/// Token rows per gr_read.cm work-group: every weight chunk is reused for all of them, so as
/// many as the SLM budget (Rbar f16 + t f32 per row) allows. Decode keeps one row per
/// work-group - the extra rows are computed anyway and only cost latency there.
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

/// Row-count regimes of the two dpas GEMM kernels. Measured on B580 (160 EU) and PTL B390
/// (96 EU) at C=4/D=2560/L=320: few rows need the K reduction split over a work-group so
/// the weights are streamed by enough threads; the split stops paying off once the token
/// tiles alone fill the GPU (~4 rows per EU for the down projection).
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
    size_t tm;      // tokens per thread
    size_t ksplit;  // threads of a work-group splitting the K reduction
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
    // Narrow D leaves few column tiles; split L until there are ~2 threads per EU.
    const size_t col_tiles = desc.hidden_size / gr_dpas_n;
    const size_t cap = std::max<size_t>(1, 2 * static_cast<size_t>(info.execution_units_count) / col_tiles);
    return {8, largest_divisor_up_to(desc.lowrank / gr_dpas_k_step, cap), 1};
}

const char* gr_tier_suffix(GrTier tier) {
    switch (tier) {
    case GrTier::decode:
        return "_decode_cm";
    case GrTier::mid:
        return "_mid_cm";
    default:
        return "_cm";
    }
}

constexpr size_t gr_read_dpas_tm = 32;  // token rows per gr_read_dpas.cm work-group

/// Above ~4 rows per EU (the down projection's wide tier) the single systolic kernel
/// gr_read_dpas.cm beats the 3-kernel pipeline (B580 3-8%, PTL ~25% at D=2560) and needs
/// no RBAR/T scratch; below it the pipeline's split-K tiers parallelize better.
bool use_single_dpas_read(const gated_residual& desc, const cldnn::device_info& info, size_t num_rows) {
    return desc.hidden_size % 32 == 0 && num_rows > 4 * static_cast<size_t>(info.execution_units_count);
}

/// Whether the systolic (cm_dpas) READ/FINAL_MIX pipeline (gr_rbar_dpas ->
/// gr_down_proj_dpas[+write gate] -> gr_up_proj_dpas[+branch mix]) can replace the single
/// fused scalar gr_read.cm kernel.
bool can_use_dpas_gr_read(const gated_residual& desc, const cldnn::device_info& info) {
    if (info.arch < gpu_arch::xe2) {
        return false;  // the kernels assert a 512-bit (16-wide) GRF
    }
    const size_t cd = desc.hc_count * desc.hidden_size;
    if (cd < 32 || cd % 32 != 0) {
        return false;  // 2D surfaces need >=64B width / 64B-aligned row pitch
    }
    if (desc.lowrank < 32 || desc.lowrank % 32 != 0) {
        return false;  // 64B T surface, 32-wide down-projection column tiles
    }
    if (desc.hc_count == 0 || desc.hc_count % 2 != 0 || desc.hc_count > gr_dpas_n) {
        return false;  // the write gate is one N16 column tile stored in half pairs
    }
    return true;
}

// ---------------------------------------------------------------------------
// G1: READ / MIX (and FINAL_MIX)
// ---------------------------------------------------------------------------
class GatedResidualReadGenerator : public KernelGenerator {
public:
    explicit GatedResidualReadGenerator(bool decode = false)
        : KernelGenerator("gr_read", decode ? "_decode_cm" : "_cm"),
          m_decode(decode) {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_gr_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();

        jit.add(make_common_jit(desc));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.add(make_jit_constant("GR_EPS", desc->epsilon));
        jit.make("GR_NORM_GAIN_PLUS_ONE", desc->norm_gain_plus_one ? 1 : 0);
        jit.make("GR_PRODUCE_WRITE_GATE", desc->produce_write_gate ? 1 : 0);
        jit.make("GR_WG_SIZE", gr_read_wg_size(params.get_device_info()));
        jit.make("GR_READ_ROWS", gr_read_rows(*desc, params.get_device_info(), m_decode));

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        const auto desc = params.typed_desc<gated_residual>();
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // residual
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // norm weight
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // W_down
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // W_up
        if (desc->produce_write_gate) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 4});  // W_inject
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // mixed
        if (desc->produce_write_gate) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});  // write gate S
        }
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // num_rows
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[decode = m_decode](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);
            const size_t wg = gr_read_wg_size(params.get_device_info());
            const size_t rows_per_wg = gr_read_rows(*desc, params.get_device_info(), decode);

            auto& wgs = kd.params.workGroups;
            wgs.global = {wg, std::max<size_t>(1, (num_rows + rows_per_wg - 1) / rows_per_wg), 1};
            wgs.local = {wg, 1, 1};

            kd.params.scalars.clear();
            scalar_desc s;
            s.t = scalar_desc::Types::UINT32;
            s.v.u32 = static_cast<uint32_t>(num_rows);
            kd.params.scalars.push_back(s);
        }};
    }

private:
    bool m_decode;
};

// ---------------------------------------------------------------------------
// G1 systolic READ / MIX in ONE kernel (many rows), see gr_read_dpas.cm
// ---------------------------------------------------------------------------
class GatedResidualReadDpasGenerator : public KernelGenerator {
public:
    GatedResidualReadDpasGenerator() : KernelGenerator("gr_read_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_gr_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        jit.add(make_common_jit(desc));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.add(make_jit_constant("GR_EPS", desc->epsilon));
        jit.make("GR_NORM_GAIN_PLUS_ONE", desc->norm_gain_plus_one ? 1 : 0);
        jit.make("GR_PRODUCE_WRITE_GATE", desc->produce_write_gate ? 1 : 0);
        jit.make("GR_WG_SIZE", gr_wg_size);
        jit.make("GR_RD_TM", gr_read_dpas_tm);
        jit.make("GR_RD_NT", 2);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        const auto desc = params.typed_desc<gated_residual>();
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // residual
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // norm weight
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // W_down
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // W_up
        if (desc->produce_write_gate) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 4});  // W_inject
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // mixed
        if (desc->produce_write_gate) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});  // write gate S
        }
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // num_rows
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);

            auto& wgs = kd.params.workGroups;
            wgs.global = {gr_wg_size, std::max<size_t>(1, (num_rows + gr_read_dpas_tm - 1) / gr_read_dpas_tm), 1};
            wgs.local = {gr_wg_size, 1, 1};

            kd.params.scalars.clear();
            scalar_desc s;
            s.t = scalar_desc::Types::UINT32;
            s.v.u32 = static_cast<uint32_t>(num_rows);
            kd.params.scalars.push_back(s);
        }};
    }
};

// ---------------------------------------------------------------------------
// G2: WRITE / COMBINE
// ---------------------------------------------------------------------------
class GatedResidualWriteGenerator : public KernelGenerator {
public:
    GatedResidualWriteGenerator() : KernelGenerator("gr_write", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_gr_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        jit.add(make_common_jit(desc));
        jit.make("GR_WRITE_TILE", gr_best_tile(desc->hidden_size));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
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
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);
            const size_t tile = gr_best_tile(desc->hidden_size);

            auto& wgs = kd.params.workGroups;
            wgs.global = {(desc->hidden_size + tile - 1) / tile, num_rows, 1};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            scalar_desc s;
            s.t = scalar_desc::Types::UINT32;
            s.v.u32 = static_cast<uint32_t>(num_rows);
            kd.params.scalars.push_back(s);
        }};
    }
};

// ---------------------------------------------------------------------------
// G1 systolic pipeline, stage 1/3: branch-wise grouped RMSNorm -> RBAR scratch
// ---------------------------------------------------------------------------
class GatedResidualRbarDpasGenerator : public KernelGenerator {
public:
    GatedResidualRbarDpasGenerator() : KernelGenerator("gr_rbar_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_gr_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        jit.add(make_common_jit(desc));
        jit.add(make_jit_constant("GR_EPS", desc->epsilon));
        jit.make("GR_NORM_GAIN_PLUS_ONE", desc->norm_gain_plus_one ? 1 : 0);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
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
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);

            auto& wgs = kd.params.workGroups;
            // One thread per (branch, row): the norm is branch-wise, so no reduction across threads.
            wgs.global = {desc->hc_count, std::max<size_t>(1, num_rows), 1};
            wgs.local = {desc->hc_count, 1, 1};

            kd.params.scalars.clear();
            scalar_desc s;
            s.t = scalar_desc::Types::UINT32;
            s.v.u32 = static_cast<uint32_t>(num_rows);
            kd.params.scalars.push_back(s);
        }};
    }
};

// ---------------------------------------------------------------------------
// G1 systolic pipeline, stage 2/3: low-rank down projection + 1/C scale + SiLU
// (+ the write-gate projection as one extra column tile)
// ---------------------------------------------------------------------------
class GatedResidualDownProjDpasGenerator : public KernelGenerator {
public:
    explicit GatedResidualDownProjDpasGenerator(GrTier tier = GrTier::wide)
        : KernelGenerator("gr_down_proj_dpas", gr_tier_suffix(tier)),
          m_tier(tier) {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_gr_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        const auto tiling = gr_down_tiling(*desc, params.get_device_info(), m_tier);
        jit.add(make_common_jit(desc));
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.make("GR_DOWN_TM", tiling.tm);
        jit.make("GR_DOWN_TN", gr_down_tn);
        jit.make("GR_DOWN_KSPLIT", tiling.ksplit);
        jit.make("GR_PRODUCE_WRITE_GATE", desc->produce_write_gate ? 1 : 0);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
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
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);
            const auto tiling = gr_down_tiling(*desc, params.get_device_info(), tier);
            const size_t col_tiles = desc->lowrank / gr_down_tn + (desc->produce_write_gate ? 1 : 0);
            const size_t row_tiles = std::max<size_t>(1, (num_rows + tiling.tm - 1) / tiling.tm);

            auto& wgs = kd.params.workGroups;
            // One work-group per (column tile, token tile); KSPLIT threads split the K reduction.
            wgs.global = {col_tiles * tiling.ksplit, row_tiles, 1};
            wgs.local = {tiling.ksplit, 1, 1};

            kd.params.scalars.clear();
            scalar_desc s;
            s.t = scalar_desc::Types::UINT32;
            s.v.u32 = static_cast<uint32_t>(num_rows);
            kd.params.scalars.push_back(s);
        }};
    }

private:
    GrTier m_tier;
};

// ---------------------------------------------------------------------------
// G1 systolic pipeline, stage 3/3: low-rank up projection + sigmoid + branch mix
// ---------------------------------------------------------------------------
class GatedResidualUpProjDpasGenerator : public KernelGenerator {
public:
    explicit GatedResidualUpProjDpasGenerator(GrTier tier = GrTier::wide)
        : KernelGenerator("gr_up_proj_dpas", gr_tier_suffix(tier)),
          m_tier(tier) {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_gr_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        const auto tiling = gr_up_tiling(*desc, params.get_device_info(), m_tier);
        jit.add(make_common_jit(desc));
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.make("GR_UP_TM", tiling.tm);
        jit.make("GR_UP_KSPLIT", tiling.ksplit);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
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
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);
            const auto tiling = gr_up_tiling(*desc, params.get_device_info(), tier);
            const size_t col_tiles = desc->hidden_size / gr_dpas_n;
            const size_t row_tiles = std::max<size_t>(1, (num_rows + tiling.tm - 1) / tiling.tm);

            auto& wgs = kd.params.workGroups;
            if (tiling.ksplit > 1) {
                wgs.global = {col_tiles * tiling.ksplit, row_tiles, 1};
                wgs.local = {tiling.ksplit, 1, 1};
            } else {
                // Work-groups of lws_y token tiles reuse the same W_up columns from L1.
                const size_t lws_y = std::min(tiling.lws_y, row_tiles);
                wgs.global = {col_tiles, (row_tiles + lws_y - 1) / lws_y * lws_y, 1};
                wgs.local = {1, lws_y, 1};
            }

            kd.params.scalars.clear();
            scalar_desc s;
            s.t = scalar_desc::Types::UINT32;
            s.v.u32 = static_cast<uint32_t>(num_rows);
            kd.params.scalars.push_back(s);
        }};
    }

private:
    GrTier m_tier;
};

class GatedResidualCMImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::GatedResidualCMImpl)

    Stage::Ptr gr_read = make_stage<GatedResidualReadGenerator>();
    Stage::Ptr gr_read_decode = make_stage<GatedResidualReadGenerator>(true);
    Stage::Ptr gr_write = make_stage<GatedResidualWriteGenerator>();
    Stage::Ptr gr_rbar_dpas = make_stage<GatedResidualRbarDpasGenerator>();
    Stage::Ptr gr_down_proj_dpas = make_stage<GatedResidualDownProjDpasGenerator>();
    Stage::Ptr gr_down_proj_dpas_mid = make_stage<GatedResidualDownProjDpasGenerator>(GrTier::mid);
    Stage::Ptr gr_down_proj_dpas_decode = make_stage<GatedResidualDownProjDpasGenerator>(GrTier::decode);
    Stage::Ptr gr_up_proj_dpas = make_stage<GatedResidualUpProjDpasGenerator>();
    Stage::Ptr gr_up_proj_dpas_decode = make_stage<GatedResidualUpProjDpasGenerator>(GrTier::decode);
    Stage::Ptr gr_read_dpas = make_stage<GatedResidualReadDpasGenerator>();

    GatedResidualCMImpl() : PrimitiveImplCM(GatedResidualCMImplementationManager::get_type_info_static()) {}
    GatedResidualCMImpl(const program_node& /*node*/, const RuntimeParams& params) : GatedResidualCMImpl() {
        const auto desc = params.typed_desc<gated_residual>();
        const auto& info = params.get_device_info();
        // READ and WRITE cannot be merged: an attention / GDN / MLP sub-layer sits between them.
        if (desc->mode == gated_residual::Mode::WRITE) {
            add_stage(gr_write, params);
        } else if (can_use_dpas_gr_read(*desc, info)) {
            // Producer/consumer order through the RBAR/T scratch buffers is enforced by
            // execute(); static shapes only build what their row count needs.
            if (params.is_dynamic()) {
                for (auto* stage : {&gr_rbar_dpas, &gr_down_proj_dpas_decode, &gr_down_proj_dpas_mid, &gr_down_proj_dpas, &gr_up_proj_dpas_decode, &gr_up_proj_dpas}) {
                    add_stage(*stage, params);
                }
                if (desc->hidden_size % 32 == 0) {
                    add_stage(gr_read_dpas, params);
                }
            } else {
                const size_t rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);
                if (use_single_dpas_read(*desc, info, rows)) {
                    add_stage(gr_read_dpas, params);
                } else {
                    add_stage(gr_rbar_dpas, params);
                    add_stage(down_stage(gr_down_tier(rows, info)), params);
                    add_stage(up_stage(gr_up_tier(rows)), params);
                }
            }
        } else if (params.is_dynamic()) {
            add_stage(gr_read_decode, params);
            add_stage(gr_read, params);
        } else {
            const size_t rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);
            add_stage(rows <= gr_decode_rows ? gr_read_decode : gr_read, params);
        }
    }

    Stage::Ptr& down_stage(GrTier tier) {
        return tier == GrTier::decode ? gr_down_proj_dpas_decode : tier == GrTier::mid ? gr_down_proj_dpas_mid : gr_down_proj_dpas;
    }

    Stage::Ptr& up_stage(GrTier tier) {
        return tier == GrTier::decode ? gr_up_proj_dpas_decode : gr_up_proj_dpas;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatedResidualCMImpl>(this);
    }

    // Default execute() (unused here) would only expose the LAST dispatched kernel's event
    // via get_executed_primitives(), which is wrong for the 3-kernel dpas READ pipeline
    // (see gr_stage_events.hpp). This override chains the stages explicitly and records
    // every stage's own event for roofline test consumption.
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
        if (has_stage(gr_rbar_dpas) || has_stage(gr_read_dpas)) {
            const auto& params = *instance.get_impl_params();
            const auto desc = params.typed_desc<gated_residual>();
            const size_t rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);
            if (has_stage(gr_read_dpas) && use_single_dpas_read(*desc, params.get_device_info(), rows)) {
                auto ev = execute_stage(events, instance, gr_read_dpas);
                record_stage_event("READ_DPAS", ev);
                return ev;
            }
            auto& down = down_stage(gr_down_tier(rows, params.get_device_info()));
            auto& up = up_stage(gr_up_tier(rows));
            OPENVINO_ASSERT(has_stage(gr_rbar_dpas) && has_stage(down) && has_stage(up), "[GPU][CM] gated_residual: dpas stage for ", rows, " rows was not built");

            auto ev_rbar = execute_stage(events, instance, gr_rbar_dpas);
            record_stage_event("RBAR", ev_rbar);
            auto ev_down = execute_stage({ev_rbar}, instance, down);
            record_stage_event("DOWN_PROJ", ev_down);
            auto ev_up = execute_stage({ev_down}, instance, up);
            record_stage_event("UP_PROJ", ev_up);
            return instance.get_network().get_stream().aggregate_events({ev_rbar, ev_down, ev_up}, true, instance.is_output());
        }
        const auto& params = *instance.get_impl_params();
        const auto desc = params.typed_desc<gated_residual>();
        const size_t rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);
        auto& read = rows <= gr_decode_rows ? gr_read_decode : gr_read;
        OPENVINO_ASSERT(has_stage(read), "[GPU][CM] gated_residual: read stage for ", rows, " rows was not built");
        auto ev = execute_stage(events, instance, read);
        record_stage_event("READ", ev);
        return ev;
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        if (params.is_dynamic()) {
            return {};
        }
        const auto desc = params.typed_desc<gated_residual>();
        const size_t cd = desc->hc_count * desc->hidden_size;
        const size_t num_rows = std::max<size_t>(1, get_num_rows(params.get_input_layout(0), cd));
        const size_t l = std::max<size_t>(1, desc->lowrank);
        if (desc->mode != gated_residual::Mode::WRITE && can_use_dpas_gr_read(*desc, params.get_device_info()) &&
            use_single_dpas_read(*desc, params.get_device_info(), num_rows)) {
            return {};  // gr_read_dpas.cm keeps its intermediates in SLM
        }

        std::vector<BufferDescriptor> bufs;
        bufs.emplace_back(num_rows * cd, ov::element::f16);  // RBAR
        bufs.emplace_back(num_rows * l, ov::element::f16);   // T_SCRATCH
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

bool GatedResidualCMImplementationManager::validate_impl(const cldnn::program_node& node) const {
    assert(node.is_type<cldnn::gated_residual>());

    if (node.has_fused_primitives()) {
        return false;
    }

    auto& engine = node.get_program().get_engine();
    const auto& config = node.get_program().get_config();
    const auto& info = engine.get_device_info();
    if (!check_cm_jit_support(engine, config) || !info.supports_immad || !config.get_use_cm()) {
        return false;
    }

    const auto desc = node.as<cldnn::gated_residual>().get_primitive();

    // First version: C == 4 only.
    if (desc->hc_count != 4) {
        return false;
    }
    if (desc->hidden_size == 0 || desc->hidden_size % 16 != 0) {
        return false;
    }
    if (desc->hc_count * desc->hidden_size > gr_max_total_width) {
        return false;
    }
    if (desc->mode != cldnn::gated_residual::Mode::WRITE) {
        if (desc->lowrank == 0 || desc->lowrank % 16 != 0 || desc->lowrank > gr_max_lowrank) {
            return false;
        }
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

std::unique_ptr<cldnn::primitive_impl> GatedResidualCMImplementationManager::create_impl(const cldnn::program_node& node,
                                                                                         const cldnn::kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::gated_residual>());
    return std::make_unique<GatedResidualCMImpl>(node, params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gated_residual)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::GatedResidualCMImpl)
