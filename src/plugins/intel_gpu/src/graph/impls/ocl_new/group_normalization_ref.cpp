// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_ref.hpp"
#include "utils/jitter.hpp"
#include "utils/kernel_base.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "openvino/core/partial_shape.hpp"
#include "primitive_ocl_base.hpp"
#include "group_normalization_inst.h"

namespace ov::intel_gpu::ocl {

namespace {

using namespace ov::intel_gpu::ocl;

void common_jit_constants(JitConstants& jit_constants, const kernel_impl_params& params) {
    auto desc = params.typed_desc<group_normalization>();
    jit_constants.make("EPSILON", static_cast<float>(desc->epsilon));
    jit_constants.make("NUM_GROUPS", desc->num_groups);
}

class GroupNormalizationGeneratorCalcMeanRef : public ov::intel_gpu::ocl::SingleKernelGenerator {
public:
    GroupNormalizationGeneratorCalcMeanRef() : SingleKernelGenerator("group_normalization_ref") {}

protected:
    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit_constants = SingleKernelGenerator::get_jit_constants(params);
        common_jit_constants(jit_constants, params);
        jit_constants.make("MEAN_KERNEL_ENABLED", 1);
        return jit_constants;
    }

    Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

        return args;
    }

    DispatchDataFunc get_dispatch_data_func() const override {
        static auto f = DISPATCH_DATA_FUNC(params, kd) {
            auto& wgs = kd.params.workGroups;

            auto desc = params.typed_desc<group_normalization>();
            auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;
            size_t num_groups = static_cast<std::size_t>(desc->num_groups);
            if (params.output_layouts[0].is_static()) {
                const auto& out_shape = params.output_layouts[0].get_shape();
                wgs.global = { out_shape[0], num_groups, 1 };
                wgs.local = { out_shape[0] *num_groups > max_wgs ? max_wgs / num_groups : out_shape[0], num_groups, 1 };
            }
        };

        return f;
    }
};

class GroupNormalizationGeneratorCalcStd : public ov::intel_gpu::ocl::SingleKernelGenerator {
public:
    GroupNormalizationGeneratorCalcStd() : SingleKernelGenerator("group_normalization_ref") {}

protected:
    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit_constants = SingleKernelGenerator::get_jit_constants(params);
        common_jit_constants(jit_constants, params);
        jit_constants.make("STANDARD_DEVIATION_KERNEL_ENABLED", 1);
        return jit_constants;
    }

    Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});

        return args;
    }

    DispatchDataFunc get_dispatch_data_func() const override {
        static auto f = DISPATCH_DATA_FUNC(params, kd) {
            auto& wgs = kd.params.workGroups;

            auto desc = params.typed_desc<group_normalization>();
            auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;
            size_t num_groups = static_cast<std::size_t>(desc->num_groups);
            if (params.output_layouts[0].is_static()) {
                const auto& out_shape = params.output_layouts[0].get_shape();
                wgs.global = { out_shape[0], num_groups, 1 };
                wgs.local = { out_shape[0] *num_groups > max_wgs ? max_wgs / num_groups : out_shape[0], num_groups, 1 };
            }
        };
        return f;
    }
};

class GroupNormalizationGeneratorNormalize : public ov::intel_gpu::ocl::SingleKernelGenerator {
public:
    GroupNormalizationGeneratorNormalize() : SingleKernelGenerator("group_normalization_ref") {}

protected:
    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit_constants = SingleKernelGenerator::get_jit_constants(params);
        common_jit_constants(jit_constants, params);
        jit_constants.make("NORMALIZE_KERNEL_ENABLED", 1);
        jit_constants.make("INPUT_INDICES_ORDER", "batch, feature, z, y, x");
        return jit_constants;
    }

    Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    DispatchDataFunc get_dispatch_data_func() const override {
        static auto f = DISPATCH_DATA_FUNC(params, kd) {
            auto& wgs = kd.params.workGroups;

            if (params.output_layouts[0].is_static()) {
                const auto& in_l = params.input_layouts[0];
                const auto& out_l = params.output_layouts[0];
                auto b = extract_channel(ChannelName::BATCH, out_l);
                auto f = extract_channel(ChannelName::FEATURE, out_l);
                auto z = extract_channel(ChannelName::Z, out_l);
                auto y = extract_channel(ChannelName::Y, out_l);
                auto x = extract_channel(ChannelName::X, out_l);

                wgs.global = { b, f * z, y * x};
                std::vector<std::vector<ChannelName>> dims_by_gws = {
                    { ChannelName::BATCH },
                    { ChannelName::FEATURE, ChannelName::Z  },
                    { ChannelName::X, ChannelName::Y }
                };

                wgs.local = get_optimal_lws(wgs.global, params.get_device_info(), in_l.format, out_l.format, dims_by_gws);
            }
        };
        return f;
    }
};

class GroupNormalizationRefImpl : public PrimitiveImplOCL {
public:
    static constexpr size_t CALC_MEAN_STAGE = 0;
    static constexpr size_t CALC_STD_STAGE = 1;
    static constexpr size_t NORMALIZE_STAGE = 2;

    GroupNormalizationRefImpl(const program_node& node, const kernel_impl_params& params)
        : PrimitiveImplOCL(std::string(GroupNormalizationRef::get_type_info_static().name)) {
        add_stage<GroupNormalizationGeneratorCalcMeanRef, CALC_MEAN_STAGE>(params);
        add_stage<GroupNormalizationGeneratorCalcStd, CALC_STD_STAGE>(params);
        add_stage<GroupNormalizationGeneratorNormalize, NORMALIZE_STAGE>(params);
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<GroupNormalizationRefImpl>(*this);
    }

    std::vector<layout> get_internal_buffer_layouts(const kernel_impl_params& params) const override {
        auto desc = params.typed_desc<group_normalization>();
        int64_t batch = params.output_layouts[0].get_shape()[0];
        auto buf = layout{ov::PartialShape{batch * desc->num_groups}, ov::element::f32, format::bfyx };
        return { buf, buf };
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GroupNormalizationRef::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<group_normalization>());
    return std::make_unique<GroupNormalizationRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::group_normalization)
