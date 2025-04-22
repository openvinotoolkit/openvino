// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_ref.hpp"

#include "common_utils/jitter.hpp"
#include "group_normalization_inst.h"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "ocl_v2/utils/fused_ops_jitter.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {

using namespace ov::intel_gpu::ocl;

void common_jit_constants(JitConstants& jit_constants, const RuntimeParams& params) {
    auto desc = params.typed_desc<group_normalization>();
    jit_constants.make("EPSILON", static_cast<float>(desc->epsilon));
    jit_constants.make("NUM_GROUPS", desc->num_groups);
}

class GroupNormalizationGeneratorCalcMeanRef : public KernelGenerator {
public:
    GroupNormalizationGeneratorCalcMeanRef() : KernelGenerator("group_normalization_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);
        common_jit_constants(jit_constants, params);
        jit_constants.make("MEAN_KERNEL_ENABLED", 1);
        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            auto desc = params.typed_desc<group_normalization>();
            auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;
            auto num_groups = static_cast<size_t>(desc->num_groups);
            const auto& out_shape = params.output_layouts[0].get_shape();
            wgs.global = {out_shape[0], num_groups, 1};
            wgs.local = {out_shape[0] * num_groups > max_wgs ? max_wgs / num_groups : out_shape[0], num_groups, 1};
        }};
    }
};

class GroupNormalizationGeneratorCalcStd : public KernelGenerator {
public:
    GroupNormalizationGeneratorCalcStd() : KernelGenerator("group_normalization_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);
        common_jit_constants(jit_constants, params);
        jit_constants.make("STANDARD_DEVIATION_KERNEL_ENABLED", 1);
        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            auto desc = params.typed_desc<group_normalization>();
            auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;
            auto num_groups = static_cast<size_t>(desc->num_groups);
            const auto& out_shape = params.output_layouts[0].get_shape();
            wgs.global = {out_shape[0], num_groups, 1};
            wgs.local = {out_shape[0] * num_groups > max_wgs ? max_wgs / num_groups : out_shape[0], num_groups, 1};
        }};
    }
};

class GroupNormalizationGeneratorNormalize : public KernelGenerator {
public:
    GroupNormalizationGeneratorNormalize() : KernelGenerator("group_normalization_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        common_jit_constants(jit, params);
        jit.make("NORMALIZE_KERNEL_ENABLED", 1);
        jit.make("INPUT_INDICES_ORDER", "batch, feature, z, y, x");

        if (params.has_fused_primitives()) {
            const auto& out_l = params.get_output_layout(0);
            auto idx_order =
                out_l.get_rank() == 5 ? std::vector<std::string>{"batch", "feature", "z", "y", "x"} : std::vector<std::string>{"batch", "feature", "y", "x"};

            FusedOpsConfiguration conf = {"", idx_order, "res", out_l.data_type};
            jit.add(make_fused_ops_jit_constants(params, {conf}));
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        add_fused_ops_arguments(args, params);

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            assert(!params.is_dynamic());

            const auto& in_l = params.input_layouts[0];
            const auto& out_l = params.output_layouts[0];
            auto b = extract_channel(ChannelName::BATCH, out_l);
            auto f = extract_channel(ChannelName::FEATURE, out_l);
            auto z = extract_channel(ChannelName::Z, out_l);
            auto y = extract_channel(ChannelName::Y, out_l);
            auto x = extract_channel(ChannelName::X, out_l);

            wgs.global = {b, f * z, y * x};
            std::vector<std::vector<ChannelName>> dims_by_gws = {{ChannelName::BATCH},
                                                                 {ChannelName::FEATURE, ChannelName::Z},
                                                                 {ChannelName::X, ChannelName::Y}};

            wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info(), in_l.format, out_l.format, dims_by_gws);
        }};
    }
};

class GroupNormalizationRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GroupNormalizationRefImpl)
    Stage::Ptr calc_mean = make_stage<GroupNormalizationGeneratorCalcMeanRef>();
    Stage::Ptr calc_mean_std = make_stage<GroupNormalizationGeneratorCalcStd>();
    Stage::Ptr normalize = make_stage<GroupNormalizationGeneratorNormalize>();

    GroupNormalizationRefImpl() : PrimitiveImplOCL(GroupNormalizationRef::get_type_info_static()) {}
    GroupNormalizationRefImpl(const program_node& node, const RuntimeParams& params) : GroupNormalizationRefImpl() {
        add_stage(calc_mean, params);
        add_stage(calc_mean_std, params);
        add_stage(normalize, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GroupNormalizationRefImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        auto desc = params.typed_desc<group_normalization>();
        size_t batch = params.output_layouts[0].get_shape()[0];
        auto buf = BufferDescriptor{batch * static_cast<size_t>(desc->num_groups), ov::element::f32};
        return {buf, buf};
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GroupNormalizationRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<group_normalization>());
    return std::make_unique<GroupNormalizationRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::group_normalization)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GroupNormalizationRefImpl)
