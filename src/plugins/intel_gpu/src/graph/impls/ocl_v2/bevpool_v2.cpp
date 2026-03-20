// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bevpool_v2.hpp"

#include "bevpool_v2_inst.h"
#include "common_utils/dispatch_utils.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class BevPoolV2Ref : public KernelGenerator {
public:
    BevPoolV2Ref() : KernelGenerator("bevpool_v2", "ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto& desc = params.typed_desc<bevpool_v2>();

        const auto inputs_count = params.input_layouts.size();
        const auto input1_length = params.is_dynamic() ? 1 : params.get_input_layout(1).count();
        const auto input3_length = params.is_dynamic() ? 1 : params.get_input_layout(3).count();

        jit.add(make_jit_constant("INPUTS_COUNT", inputs_count));
        jit.add(make_jit_constant("INPUT_CHANNELS", desc->input_channels));
        jit.add(make_jit_constant("OUTPUT_CHANNELS", desc->output_channels));
        jit.add(make_jit_constant("IMAGE_WIDTH", desc->image_width));
        jit.add(make_jit_constant("IMAGE_HEIGHT", desc->image_height));
        jit.add(make_jit_constant("FEATURE_WIDTH", desc->feature_width));
        jit.add(make_jit_constant("FEATURE_HEIGHT", desc->feature_height));
        jit.add(make_jit_constant("D_BOUND_MIN", desc->d_bound.min));
        jit.add(make_jit_constant("D_BOUND_MAX", desc->d_bound.max));
        jit.add(make_jit_constant("D_BOUND_STEP", desc->d_bound.step));
        jit.add(make_jit_constant("INPUT1_LENGTH", input1_length));
        jit.add(make_jit_constant("INPUT3_LENGTH", input3_length));

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            const auto& desc = params.typed_desc<bevpool_v2>();

            const size_t output_channels = static_cast<size_t>(desc->output_channels);
            const size_t interval_count = params.is_dynamic() ? 1 : static_cast<size_t>(params.get_input_layout(3).count() / 3);

            const size_t gws0 = ((output_channels + 15) / 16) * 16;
            wgs.global = {gws0, std::max(interval_count, static_cast<size_t>(1)), 1};
            wgs.local = {std::min(gws0, static_cast<size_t>(16)), 1, 1};
        }};
    }
};

class BevPoolV2Impl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::BevPoolV2Impl)

    Stage::Ptr ref_stage = make_stage<BevPoolV2Ref>();

    BevPoolV2Impl() : PrimitiveImplOCL(BevPoolV2::get_type_info_static()) {}
    BevPoolV2Impl(const program_node& node, const RuntimeParams& params) : BevPoolV2Impl() {
        add_stage(ref_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<BevPoolV2Impl>(this);
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        cldnn::stream& stream = instance.get_network().get_stream();
        stream.enqueue_barrier();

        auto output_evt = instance.output_memory_ptr(0)->fill(stream, false);
        std::vector<cldnn::event::ptr> deps(events);
        deps.push_back(output_evt);

        return PrimitiveImplOCL::execute(deps, instance);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> BevPoolV2::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<bevpool_v2>());
    return std::make_unique<BevPoolV2Impl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::bevpool_v2)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::BevPoolV2Impl)
