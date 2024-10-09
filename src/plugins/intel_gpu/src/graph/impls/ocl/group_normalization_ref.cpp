// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_ref.hpp"
#include "impls/ocl/kernel_base.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "openvino/core/partial_shape.hpp"
#include "primitive_ocl_base.hpp"
#include "group_normalization_inst.h"

namespace cldnn {
namespace ocl {

namespace {

using namespace ov::intel_gpu::ocl;

class GroupNormalizationGeneratorCalcMeanRef : public ov::intel_gpu::ocl::SingleKernelGenerator {
public:
    GroupNormalizationGeneratorCalcMeanRef() : SingleKernelGenerator("group_normalization_gpu_ref") {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit_constants = SingleKernelGenerator::get_jit_constants(node, params);
        jit_constants.make("MEAN_KERNEL_ENABLED", 1);
        return jit_constants;
    }

    Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

        return args;
    }

    DispatchDataFunc get_dispatch_data_func(const kernel_impl_params& params) const override {
        static auto f = DISPATCH_DATA_FUNC(params) {
            WorkGroupSizes dispatch_data;

            auto desc = params.typed_desc<group_normalization>();
            auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;
            size_t num_groups = static_cast<std::size_t>(desc->num_groups);
            if (params.output_layouts[0].is_static()) {
                const auto& out_shape = params.output_layouts[0].get_shape();
                dispatch_data.global = { out_shape[0], num_groups, 1 };
                dispatch_data.local = { out_shape[0] *num_groups > max_wgs ? max_wgs / num_groups : out_shape[0], num_groups, 1 };
            }
            return { dispatch_data, {} };
        };

        return f;
    }

    std::vector<layout> get_interanl_buffers(const program_node& node, const kernel_impl_params& params) const override {
        auto desc = params.typed_desc<group_normalization>();
        int64_t batch = params.output_layouts[0].get_shape()[0];
        return { layout{ov::PartialShape{batch * desc->num_groups}, ov::element::f32, format::bfyx } };
    }
};

class GroupNormalizationGeneratorCalcStd : public ov::intel_gpu::ocl::SingleKernelGenerator {
public:
    GroupNormalizationGeneratorCalcStd() : SingleKernelGenerator("group_normalization_gpu_ref") {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit_constants = SingleKernelGenerator::get_jit_constants(node, params);
        jit_constants.make("STANDARD_DEVIATION_KERNEL_ENABLED", 1);
        return jit_constants;
    }

    Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});

        return args;
    }

    DispatchDataFunc get_dispatch_data_func(const kernel_impl_params& params) const override {
        static auto f = DISPATCH_DATA_FUNC(params) {
            WorkGroupSizes dispatch_data;

            auto desc = params.typed_desc<group_normalization>();
            auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;
            size_t num_groups = static_cast<std::size_t>(desc->num_groups);
            if (params.output_layouts[0].is_static()) {
                const auto& out_shape = params.output_layouts[0].get_shape();
                dispatch_data.global = { out_shape[0], num_groups, 1 };
                dispatch_data.local = { out_shape[0] *num_groups > max_wgs ? max_wgs / num_groups : out_shape[0], num_groups, 1 };
            }
            return { dispatch_data, {} };
        };
        return f;
    }

    std::vector<layout> get_interanl_buffers(const program_node& node, const kernel_impl_params& params) const override {
        auto desc = params.typed_desc<group_normalization>();
        int64_t batch = params.output_layouts[0].get_shape()[0];
        return { layout{ov::PartialShape{batch * desc->num_groups}, ov::element::f32, format::bfyx } };
    }
};

class GroupNormalizationGeneratorNormalize : public ov::intel_gpu::ocl::SingleKernelGenerator {
public:
    GroupNormalizationGeneratorNormalize() : SingleKernelGenerator("group_normalization_gpu_ref") {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit_constants = SingleKernelGenerator::get_jit_constants(node, params);
        jit_constants.make("NORMALIZE_KERNEL_ENABLED", 1);
        jit_constants.make("INPUT_INDICES_ORDER", "batch, feature, z, y, x");
        return jit_constants;
    }

    Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    DispatchDataFunc get_dispatch_data_func(const kernel_impl_params& params) const override {
        static auto f = DISPATCH_DATA_FUNC(params) {
            WorkGroupSizes dispatch_data;

            if (params.output_layouts[0].is_static()) {
                const auto& out_shape = params.output_layouts[0].get_shape();
                if (out_shape.size() == 4) {
                    dispatch_data.global = { out_shape[0], out_shape[1], out_shape[2] * out_shape[3]};
                } else if (out_shape.size() == 4) {
                    dispatch_data.global = { out_shape[0], out_shape[1] * out_shape[2], out_shape[3] * out_shape[4]};
                }
                dispatch_data.local = {1, 1, 1}; //GetOptimalLocalWorkGroupSizes(dispatch_data.global, params.engineInfo, in_layout, out_layout, dims_by_gws);
            }
            return { dispatch_data, {} };
        };
        return f;
    }
};

class GroupNormalizationGeneratorRef : public ov::intel_gpu::ocl::MultiStageKernelGenerator {
public:
    GroupNormalizationGeneratorRef() : MultiStageKernelGenerator(
            GroupNormalizationGeneratorCalcMeanRef(),
            GroupNormalizationGeneratorCalcStd(),
            GroupNormalizationGeneratorNormalize()
    ) {}

    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        JitConstants jit_constants;
        auto desc = params.typed_desc<group_normalization>();
        jit_constants.make("EPSILON", static_cast<float>(desc->epsilon));
        jit_constants.make("NUM_GROUPS", desc->num_groups);
        return jit_constants;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GroupNormalizationRef::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<group_normalization>());
    GroupNormalizationGeneratorRef gen;
    return cldnn::make_unique<primitive_impl_ocl>(gen.get_kernels_data(node, params), std::string(get_type_info().name));
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::group_normalization)
