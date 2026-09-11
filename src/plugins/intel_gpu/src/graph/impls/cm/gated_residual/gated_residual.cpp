// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_residual.hpp"

#include <memory>

#include "../primitive_cm_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/gated_residual.hpp"
#include "registry/implementation_manager.hpp"

namespace ov::intel_gpu::cm {

using namespace cldnn;

namespace {

constexpr size_t gr_wg_size = 16;
constexpr size_t gr_max_lowrank = 512;      // GR_LOWRANK is kept in registers by gr_read.cm
constexpr size_t gr_max_total_width = 16384;  // C*D must fit into SLM as f16

constexpr auto get_gr_build_options() {
    return " -cmc -Qxcm_register_file_size=256";
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
            if (params.is_dynamic()) {
                return;
            }
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
        jit.add(make_common_jit(params.typed_desc<gated_residual>()));
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
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<gated_residual>();
            const size_t num_rows = get_num_rows(params.get_input_layout(0), desc->hc_count * desc->hidden_size);

            auto& wgs = kd.params.workGroups;
            wgs.global = {desc->hidden_size / 16, desc->hc_count, num_rows};
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

    GatedResidualCMImpl() : PrimitiveImplCM(GatedResidualCMImplementationManager::get_type_info_static()) {}
    GatedResidualCMImpl(const program_node& /*node*/, const RuntimeParams& params) : GatedResidualCMImpl() {
        const auto desc = params.typed_desc<gated_residual>();
        // READ and WRITE cannot be merged: an attention / GDN / MLP sub-layer sits between them.
        if (desc->mode == gated_residual::Mode::WRITE) {
            add_stage(gr_write, params);
        } else {
            add_stage(gr_read, params);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatedResidualCMImpl>(this);
    }
};

}  // namespace

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
