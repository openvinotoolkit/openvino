// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bevpool_v2.hpp"

#include "bevpool_v2_inst.h"
#include "common_utils/dispatch_utils.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

#include <algorithm>
#include <cstdlib>

namespace ov::intel_gpu::ocl {
namespace {

#ifndef OV_GPU_BEVPOOL_V2_ENABLE_OPT_BLOCK8
#define OV_GPU_BEVPOOL_V2_ENABLE_OPT_BLOCK8 1
#endif

#ifndef OV_GPU_BEVPOOL_V2_ENABLE_OPT_BLOCK4
#define OV_GPU_BEVPOOL_V2_ENABLE_OPT_BLOCK4 1
#endif

enum KernelsTypes {
    REF = 0,
    OPT4,
    OPT8,
};

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

template <size_t BlockSize>
class BevPoolV2Opt : public KernelGenerator {
public:
    BevPoolV2Opt() : KernelGenerator("bevpool_v2_opt", BlockSize == 8 ? "opt8" : "opt4") {}

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
        jit.add(make_jit_constant("BLOCK_SIZE", BlockSize));

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

            const size_t gws0 = (output_channels + BlockSize - 1) / BlockSize;
            wgs.global = {std::max(gws0, static_cast<size_t>(1)), std::max(interval_count, static_cast<size_t>(1)), 1};
            wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info());
        }};
    }
};

template <size_t BlockSize>
bool support_opt_kernel(const kernel_impl_params& params) {
    if (params.is_dynamic())
        return false;

    const auto& desc = params.typed_desc<bevpool_v2>();
    const size_t output_channels = static_cast<size_t>(desc->output_channels);
    const size_t interval_count = static_cast<size_t>(params.get_input_layout(3).count() / 3);
    const bool is_fp16_input = params.get_input_layout(0).data_type == ov::element::f16;
    const auto& info = params.get_device_info();

    const bool has_subgroup_support = info.supports_khr_subgroups || info.supports_intel_subgroups;
    if (!has_subgroup_support)
        return false;

    if (info.max_work_group_size < BlockSize)
        return false;

    if (!info.supported_simd_sizes.empty()) {
        const bool has_required_simd = std::find(info.supported_simd_sizes.begin(),
                                                 info.supported_simd_sizes.end(),
                                                 BlockSize) != info.supported_simd_sizes.end();
        if (!has_required_simd)
            return false;
    }

    // P3 specialization: use stricter rules for fp16 block8 fast path to reduce tail overhead.
    if constexpr (BlockSize == 8) {
        if (is_fp16_input) {
            if ((output_channels % BlockSize) != 0 || interval_count < 16)
                return false;
        }
    }

    // First-pass heuristic: use blocked kernels when channels and interval workload are large enough.
    if (output_channels < BlockSize || interval_count < 8)
        return false;

    return true;
}

bool env_enabled(const char* name) {
    if (const char* value = std::getenv(name)) {
        return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' || value[0] == 't' || value[0] == 'T';
    }
    return false;
}

class BevPoolV2Impl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::BevPoolV2Impl)

    Stage::Ptr ref_stage = make_stage<BevPoolV2Ref>();
    Stage::Ptr opt4_stage = make_stage<BevPoolV2Opt<4>>();
    Stage::Ptr opt8_stage = make_stage<BevPoolV2Opt<8>>();

    BevPoolV2Impl() : PrimitiveImplOCL(BevPoolV2::get_type_info_static()) {}
    BevPoolV2Impl(const program_node& node, const RuntimeParams& params) : BevPoolV2Impl() {
        add_stage(ref_stage, params);
        add_stage(opt4_stage, params);
        add_stage(opt8_stage, params);
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

    std::vector<size_t> get_stages_execution_order(const cldnn::primitive_inst& instance) const override {
        auto params = instance.get_impl_params();

        // P5 helper switches for deterministic benchmarking.
        if (env_enabled("OV_GPU_BEVPOOL_V2_FORCE_REF"))
            return {KernelsTypes::REF};
        if (env_enabled("OV_GPU_BEVPOOL_V2_FORCE_OPT8") && support_opt_kernel<8>(*params))
            return {KernelsTypes::OPT8};
        if (env_enabled("OV_GPU_BEVPOOL_V2_FORCE_OPT4") && support_opt_kernel<4>(*params))
            return {KernelsTypes::OPT4};

#if OV_GPU_BEVPOOL_V2_ENABLE_OPT_BLOCK8
        if (support_opt_kernel<8>(*params)) {
            return {KernelsTypes::OPT8};
        }
#endif
#if OV_GPU_BEVPOOL_V2_ENABLE_OPT_BLOCK4
        if (support_opt_kernel<4>(*params)) {
            return {KernelsTypes::OPT4};
        }
#endif
        return {KernelsTypes::REF};
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
