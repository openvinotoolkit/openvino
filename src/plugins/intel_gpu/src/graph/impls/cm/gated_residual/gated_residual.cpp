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
constexpr size_t gr_dpas_m = 8;               // dpas RepeatCount (M axis) used by the down/up projections
constexpr size_t gr_dpas_n = 16;              // dpas N axis == GRF width on Xe2

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
/// both gr_write.cm (branch loop moved inside the kernel) and gr_combine.cm; 16 is always
/// a valid fallback since GR_HIDDEN % 16 == 0 is enforced by validate_impl.
size_t gr_best_tile(size_t width) {
    for (size_t t : {size_t{64}, size_t{32}, size_t{16}}) {
        if (width % t == 0) {
            return t;
        }
    }
    return 16;
}

/// Whether the systolic (cm_dpas) READ/FINAL_MIX pipeline (gr_rbar_dpas -> gr_down_proj_dpas
/// -> gr_up_proj_dpas -> [gr_writegate_dpas] -> gr_combine) can replace the single fused
/// scalar gr_read.cm kernel. Mirrors qsa.cpp's can_use_dpas_projection: the down/up
/// projections are the identical M8/N16/K16 systolic GEMM shape as qsa_project_dpas.cm.
bool can_use_dpas_gr_read(const gated_residual& desc, const cldnn::device_info& info) {
    if (info.arch < gpu_arch::xe2) {
        return false;  // the kernels assert a 512-bit (16-wide) GRF
    }
    const size_t cd = desc.hc_count * desc.hidden_size;
    if (cd < 32 || cd % 32 != 0) {
        return false;  // 2D surfaces need >=64B width / 64B-aligned row pitch
    }
    if (desc.lowrank < 32 || desc.lowrank % 32 != 0) {
        return false;
    }
    if (desc.hc_count == 0 || desc.hc_count % 2 != 0 || desc.hc_count > 8) {
        return false;  // write-gate dpas RepeatCount must fit one tile and pack in half-pairs
    }
    return true;
}

// ---------------------------------------------------------------------------
// G1: READ / MIX (and FINAL_MIX)
// ---------------------------------------------------------------------------
class GatedResidualReadGenerator : public KernelGenerator {
public:
    GatedResidualReadGenerator() : KernelGenerator("gr_read", "_cm") {}

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
            wgs.global = {gr_wg_size, num_rows, 1};
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
// G1 systolic pipeline, stage 1/4: branch-wise grouped RMSNorm -> RBAR scratch
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
        jit.make("GR_WG_SIZE", gr_wg_size);
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
            wgs.global = {gr_wg_size, num_rows, 1};
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
// G1 systolic pipeline, stage 2/4: low-rank down projection + 1/C scale + SiLU
// ---------------------------------------------------------------------------
class GatedResidualDownProjDpasGenerator : public KernelGenerator {
public:
    GatedResidualDownProjDpasGenerator() : KernelGenerator("gr_down_proj_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_gr_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        jit.add(make_common_jit(desc));
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::RBAR});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // W_down
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::T_SCRATCH});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // num_rows
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);

            auto& wgs = kd.params.workGroups;
            // gws = [L/8, ceil(rows/16), 1]; one work-item owns an M8 x N16 dpas tile.
            wgs.global = {desc->lowrank / gr_dpas_m, (num_rows + gr_dpas_n - 1) / gr_dpas_n, 1};
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
// G1 systolic pipeline, stage 3/4: low-rank up projection (read-gate logits)
// ---------------------------------------------------------------------------
class GatedResidualUpProjDpasGenerator : public KernelGenerator {
public:
    GatedResidualUpProjDpasGenerator() : KernelGenerator("gr_up_proj_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_gr_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        jit.add(make_common_jit(desc));
        jit.make("GR_LOWRANK", desc->lowrank);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::T_SCRATCH});
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // W_up
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::GATE_SCRATCH});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // num_rows
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);
            const size_t cd = desc->hc_count * desc->hidden_size;

            auto& wgs = kd.params.workGroups;
            // gws = [C*D/8, ceil(rows/16), 1]; one work-item owns an M8 x N16 dpas tile.
            wgs.global = {cd / gr_dpas_m, (num_rows + gr_dpas_n - 1) / gr_dpas_n, 1};
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
// G1 systolic pipeline, stage 4a/4 (optional): write-gate low-rank projection
// ---------------------------------------------------------------------------
class GatedResidualWriteGateDpasGenerator : public KernelGenerator {
public:
    GatedResidualWriteGateDpasGenerator() : KernelGenerator("gr_writegate_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_gr_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_common_jit(params.typed_desc<gated_residual>()));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::RBAR});
        args.push_back({ArgumentDescriptor::Types::INPUT, 4});   // W_inject
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});  // write gate S
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // num_rows
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);

            auto& wgs = kd.params.workGroups;
            // gws = [1, ceil(rows/16), 1]; a single dpas tile covers all C(<=8) branches.
            wgs.global = {1, (num_rows + gr_dpas_n - 1) / gr_dpas_n, 1};
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
// G1 systolic pipeline, stage 4b/4: branch reduce-mean combine (no dpas)
// ---------------------------------------------------------------------------
class GatedResidualCombineGenerator : public KernelGenerator {
public:
    GatedResidualCombineGenerator() : KernelGenerator("gr_combine", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_gr_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<gated_residual>();
        jit.add(make_common_jit(desc));
        jit.make("GR_COMBINE_TILE", gr_best_tile(desc->hidden_size));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::RBAR});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, GatedResidualInternalBufIdx::GATE_SCRATCH});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // mixed
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

class GatedResidualCMImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::GatedResidualCMImpl)

    Stage::Ptr gr_read = make_stage<GatedResidualReadGenerator>();
    Stage::Ptr gr_write = make_stage<GatedResidualWriteGenerator>();
    Stage::Ptr gr_rbar_dpas = make_stage<GatedResidualRbarDpasGenerator>();
    Stage::Ptr gr_down_proj_dpas = make_stage<GatedResidualDownProjDpasGenerator>();
    Stage::Ptr gr_up_proj_dpas = make_stage<GatedResidualUpProjDpasGenerator>();
    Stage::Ptr gr_writegate_dpas = make_stage<GatedResidualWriteGateDpasGenerator>();
    Stage::Ptr gr_combine = make_stage<GatedResidualCombineGenerator>();

    GatedResidualCMImpl() : PrimitiveImplCM(GatedResidualCMImplementationManager::get_type_info_static()) {}
    GatedResidualCMImpl(const program_node& /*node*/, const RuntimeParams& params) : GatedResidualCMImpl() {
        const auto desc = params.typed_desc<gated_residual>();
        // READ and WRITE cannot be merged: an attention / GDN / MLP sub-layer sits between them.
        if (desc->mode == gated_residual::Mode::WRITE) {
            add_stage(gr_write, params);
        } else if (can_use_dpas_gr_read(*desc, params.get_device_info())) {
            // Default execute() chains added stages strictly in registration order (each
            // stage waits on the previous stage's event before dispatching), which is
            // exactly the safe producer/consumer ordering the RBAR/T/gate scratch round
            // trips through global memory need - see gr_rbar_dpas.cm.
            add_stage(gr_rbar_dpas, params);
            add_stage(gr_down_proj_dpas, params);
            add_stage(gr_up_proj_dpas, params);
            if (desc->produce_write_gate) {
                add_stage(gr_writegate_dpas, params);
            }
            add_stage(gr_combine, params);
        } else {
            add_stage(gr_read, params);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatedResidualCMImpl>(this);
    }

    // Default execute() (unused here) would only expose the LAST dispatched kernel's event
    // via get_executed_primitives(), which is wrong for the 4-5-kernel dpas READ pipeline
    // (see gr_stage_events.hpp). This override reproduces the same stage ordering/event
    // chaining the default implementation would have done, plus records every stage's own
    // event for roofline test consumption.
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
        if (has_stage(gr_rbar_dpas)) {
            auto ev_rbar = execute_stage(events, instance, gr_rbar_dpas);
            record_stage_event("RBAR", ev_rbar);
            auto ev_down = execute_stage({ev_rbar}, instance, gr_down_proj_dpas);
            record_stage_event("DOWN_PROJ", ev_down);
            auto ev_up = execute_stage({ev_down}, instance, gr_up_proj_dpas);
            record_stage_event("UP_PROJ", ev_up);
            std::vector<event::ptr> all_events = {ev_rbar, ev_down, ev_up};
            event::ptr prev = ev_up;
            if (has_stage(gr_writegate_dpas)) {
                auto ev_wg = execute_stage({prev}, instance, gr_writegate_dpas);
                record_stage_event("WRITE_GATE", ev_wg);
                all_events.push_back(ev_wg);
                prev = ev_wg;
            }
            auto ev_combine = execute_stage({prev}, instance, gr_combine);
            record_stage_event("COMBINE", ev_combine);
            all_events.push_back(ev_combine);
            return instance.get_network().get_stream().aggregate_events(all_events, true, instance.is_output());
        }
        auto ev = execute_stage(events, instance, gr_read);
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

        std::vector<BufferDescriptor> bufs;
        bufs.emplace_back(num_rows * cd, ov::element::f16);  // RBAR
        bufs.emplace_back(num_rows * l, ov::element::f16);   // T_SCRATCH
        bufs.emplace_back(num_rows * cd, ov::element::f16);  // GATE_SCRATCH
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
