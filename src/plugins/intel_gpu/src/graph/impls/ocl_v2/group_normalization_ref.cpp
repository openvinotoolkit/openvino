// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_ref.hpp"

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#include "common_utils/dispatch_utils.hpp"
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

// Work-group size of calc_stats. One work-group reduces one (batch, group) pair.
// It does not depend on the shape, so the same value is used for static and dynamic kernels.
size_t get_calc_stats_lws(const RuntimeParams& params) {
    const size_t max_wgs = params.get_device_info().max_work_group_size;
    size_t lws = 256;
    while (lws > 1 && lws > max_wgs) {
        lws /= 2;
    }
    return lws;
}

class GroupNormalizationGeneratorRefBase : public KernelGenerator {
public:
    explicit GroupNormalizationGeneratorRefBase(std::string_view suffix) : KernelGenerator("group_normalization_ref", suffix) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<group_normalization>();
        jit.make("EPSILON", static_cast<float>(desc->epsilon));
        jit.make("NUM_GROUPS", desc->num_groups);
        return jit;
    }
};

class GroupNormalizationGeneratorCalcStats : public GroupNormalizationGeneratorRefBase {
public:
    GroupNormalizationGeneratorCalcStats() : GroupNormalizationGeneratorRefBase("calc_stats") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GroupNormalizationGeneratorRefBase::get_jit_constants(params);
        jit.make("GROUP_NORM_KERNEL_CALC_STATS", 1);
        jit.make("LWS", get_calc_stats_lws(params));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const auto& in_l = params.input_layouts[0];
            const auto num_groups = static_cast<size_t>(params.typed_desc<group_normalization>()->num_groups);
            const auto b = extract_channel(ChannelName::BATCH, in_l);
            const auto f = extract_channel(ChannelName::FEATURE, in_l);
            OPENVINO_ASSERT(num_groups > 0 && f % num_groups == 0, "[GPU] GroupNormalization: ", f, " channels are not divisible by ", num_groups, " groups");

            const auto lws = get_calc_stats_lws(params);
            wgs.global = {lws * num_groups, std::max<size_t>(b, 1), 1};
            wgs.local = {lws, 1, 1};
        }};
    }
};

class GroupNormalizationGeneratorNormalize : public GroupNormalizationGeneratorRefBase {
public:
    GroupNormalizationGeneratorNormalize() : GroupNormalizationGeneratorRefBase("normalize") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GroupNormalizationGeneratorRefBase::get_jit_constants(params);
        jit.make("GROUP_NORM_KERNEL_NORMALIZE", 1);

        if (params.has_fused_primitives()) {
            const auto rank = params.get_output_layout(0).get_rank();
            std::vector<std::string> idx_order;
            if (rank == 6) {
                idx_order = {"b", "f", "w", "z", "y", "x"};
            } else if (rank == 5) {
                idx_order = {"b", "f", "z", "y", "x"};
            } else {
                idx_order = {"b", "f", "y", "x"};
            }

            FusedOpsConfiguration conf = {"", idx_order, "normalized", ov::element::f32, 1};
            jit.add(make_fused_ops_jit_constants(params, {conf}));
        }

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
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        add_fused_ops_arguments(args, params);

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const auto& in_l = params.input_layouts[0];
            const auto& out_l = params.output_layouts[0];
            const auto b = extract_channel(ChannelName::BATCH, out_l);
            const auto f = extract_channel(ChannelName::FEATURE, out_l);
            const auto spatial = extract_channel(ChannelName::W, out_l) * extract_channel(ChannelName::Z, out_l) * extract_channel(ChannelName::Y, out_l) *
                                 extract_channel(ChannelName::X, out_l);

            // get_optimal_lws returns lws that divides gws on every axis, so gws must not be zero.
            wgs.global = {std::max<size_t>(spatial, 1), std::max<size_t>(f, 1), std::max<size_t>(b, 1)};
            std::vector<std::vector<ChannelName>> dims_by_gws = {{ChannelName::X, ChannelName::Y, ChannelName::Z, ChannelName::W},
                                                                 {ChannelName::FEATURE},
                                                                 {ChannelName::BATCH}};
            wgs.local = get_optimal_lws(wgs.global, params.get_device_info(), in_l.format, out_l.format, dims_by_gws);
        }};
    }
};

class GroupNormalizationRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GroupNormalizationRefImpl)
    Stage::Ptr calc_stats = make_stage<GroupNormalizationGeneratorCalcStats>();
    Stage::Ptr normalize = make_stage<GroupNormalizationGeneratorNormalize>();

    GroupNormalizationRefImpl() : PrimitiveImplOCL(GroupNormalizationRef::get_type_info_static()) {}
    GroupNormalizationRefImpl(const program_node& node, const RuntimeParams& params) : GroupNormalizationRefImpl() {
        add_stage(calc_stats, params);
        add_stage(normalize, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GroupNormalizationRefImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        auto desc = params.typed_desc<group_normalization>();
        // Use get_max_shape() for upper bounded dynamic shape. This is not called for non upper bounded dynamic shape.
        // Batch is at least 1 because calc_stats runs one batch even when the batch is empty.
        const size_t batch = std::max<size_t>(params.output_layouts[0].get_partial_shape().get_max_shape()[0], 1);
        // mean and 1 / sqrt(variance + epsilon) per (batch, group)
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
