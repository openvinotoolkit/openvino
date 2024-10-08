// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_ref.hpp"
#include "impls/ocl/kernel_base.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "primitive_ocl_base.hpp"
#include "group_normalization_inst.h"

namespace cldnn {
namespace ocl {

namespace {

using namespace ov::intel_gpu::ocl;

class GroupNormalizationGeneratorCalcMeanRef : public ov::intel_gpu::ocl::SingleKernelGenerator {
public:
    GroupNormalizationGeneratorCalcMeanRef() : SingleKernelGenerator("") {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit_constants = make_base_jit_constants(node, params);
        jit_constants.make("MEAN_KERNEL_ENABLED", 1);
        return jit_constants;
    }

    Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const override {
        Arguments args;

        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }

        return args;
    }

    WorkGroupSizes get_dispatch_data(const program_node& node, const kernel_impl_params& params) const override {
        WorkGroupSizes dispatch_data;
        const auto& output = params.output_layouts[0];

        dispatch_data.global = {output.get_shape()[0], 1, 1};
        dispatch_data.local = {1, 1, 1}; /*GetOptimalLocalWorkGroupSizes(dispatch_data.gws, kernel_params.engineInfo)*/;
        return dispatch_data;
    }
};

class GroupNormalizationGeneratorCalcStd : public ov::intel_gpu::ocl::SingleKernelGenerator {
public:
    GroupNormalizationGeneratorCalcStd() : SingleKernelGenerator("") {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit_constants = make_base_jit_constants(node, params);
        jit_constants.make("STANDARD_DEVIATION_KERNEL_ENABLED", 1);
        return jit_constants;
    }

    Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const override {
        Arguments args;

        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }

        return args;
    }

    WorkGroupSizes get_dispatch_data(const program_node& node, const kernel_impl_params& params) const override {
        WorkGroupSizes dispatch_data;
        const auto& output = params.output_layouts[0];

        dispatch_data.global = {output.get_shape()[0], 1, 1};
        dispatch_data.local = {1, 1, 1}; /*GetOptimalLocalWorkGroupSizes(dispatch_data.gws, kernel_params.engineInfo)*/;
        return dispatch_data;
    }
};

class GroupNormalizationGeneratorNormalize : public ov::intel_gpu::ocl::SingleKernelGenerator {
public:
    GroupNormalizationGeneratorNormalize() : SingleKernelGenerator("") {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit_constants = make_base_jit_constants(node, params);
        jit_constants.make("STANDARD_DEVIATION_KERNEL_ENABLED", 1);
        return jit_constants;
    }

    Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const override {
        Arguments args;

        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }

        return args;
    }

    WorkGroupSizes get_dispatch_data(const program_node& node, const kernel_impl_params& params) const override {
        WorkGroupSizes dispatch_data;
        const auto& output = params.output_layouts[0];

        dispatch_data.global = {output.get_shape()[0], 1, 1};
        dispatch_data.local = {1, 1, 1}; /*GetOptimalLocalWorkGroupSizes(dispatch_data.gws, kernel_params.engineInfo)*/;
        return dispatch_data;
    }
};

class GroupNormalizationGeneratorRef : public ov::intel_gpu::ocl::MultiKernelGenerator {
public:
    GroupNormalizationGeneratorRef() : MultiKernelGenerator(
            GroupNormalizationGeneratorCalcMeanRef(),
            GroupNormalizationGeneratorCalcStd(),
            GroupNormalizationGeneratorNormalize()
    ) {}
};

}  // namespace

std::unique_ptr<primitive_impl> GroupNormalizationRef::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<group_normalization>());
    GroupNormalizationGeneratorRef gen;
    auto kd = gen.get_kernels_data(node, params);
    return cldnn::make_unique<primitive_impl_ocl>(kd, std::string(get_type_info().name));
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::group_normalization)
