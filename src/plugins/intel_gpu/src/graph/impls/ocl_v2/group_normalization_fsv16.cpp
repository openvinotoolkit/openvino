// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_fsv16.hpp"

#include "common_utils/jitter.hpp"
#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "ocl_v2/utils/fused_ops_jitter.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

constexpr size_t fsv = 16;
constexpr size_t simd = fsv;

ov::element::Type get_activation_type(const RuntimeParams& params) {
    if (params.get_input_layout(0).data_type == ov::element::f16) {
        return ov::element::f16;
    }
    return ov::element::f32;
}

ov::element::Type get_accumulator_type(const RuntimeParams& params) {
    auto in_dt = params.get_input_layout(0).data_type;
    switch (in_dt) {
    case ov::element::u8:
    case ov::element::i8:
        return ov::element::i32;
    default:
        return ov::element::f32;
    }
}

class GroupNormalizationGeneratorBase : public KernelGenerator {
public:
    explicit GroupNormalizationGeneratorBase(std::string_view name, std::string_view suffix) : KernelGenerator(name, suffix) {}
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<group_normalization>();
        jit.make("EPSILON", static_cast<float>(desc->epsilon));
        jit.make("NUM_GROUPS", desc->num_groups);
        jit.make("SIMD", simd);
        jit.make("FSV", fsv);

        if (params.is_dynamic()) {
            jit.make("GWS0", "get_global_size(0)");
            jit.make("LWS0", "get_local_size(0)");
            jit.make("SLM_SIZE", params.get_device_info().max_work_group_size);
        } else {
            KernelData kd;
            get_dispatch_data_func()(params, kd);
            const auto& wgs = kd.params.workGroups;

            jit.make("GWS0", wgs.global[0]);
            jit.make("LWS0", wgs.local[0]);
            jit.make("SLM_SIZE", wgs.local[0]);
        }

        auto activation_type = get_activation_type(params);
        jit.add(make_type_jit_constants("ACTIVATION", activation_type));
        jit.add(make_type_jit_constants("ACCUMULATOR", get_accumulator_type(params)));

        return jit;
    }
};

class GroupNormalizationGeneratorCalcSQRMean : public GroupNormalizationGeneratorBase {
public:
    GroupNormalizationGeneratorCalcSQRMean() : GroupNormalizationGeneratorBase("group_normalization_fsv16", "calc_sqr_mean") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GroupNormalizationGeneratorBase::get_jit_constants(params);
        jit.make("GROUP_NORM_KERNEL_FEATURE_MEAN_SQR_MEAN", 1);
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

            const auto& ol = params.output_layouts[0];
            auto x = extract_channel(ChannelName::X, ol);
            auto y = extract_channel(ChannelName::Y, ol);
            auto f = extract_channel(ChannelName::FEATURE, ol);
            auto b = extract_channel(ChannelName::BATCH, ol);

            wgs.global[0] = x * y;
            wgs.global[1] = ceil_div(f, fsv) * b;
            wgs.global[2] = 1;

            wgs.local[0] = x * y;
            wgs.local[1] = 1;
            wgs.local[2] = 1;

            auto max_wgs = params.get_device_info().max_work_group_size;

            size_t divisor = 2;
            while (wgs.local[0] > (max_wgs / fsv)) {
                if (wgs.global[0] % divisor == 0) {
                    wgs.local[0] = wgs.global[0] / divisor;
                }
                divisor += 1;
            }
            wgs.local[0] *= fsv;
            wgs.global[0] = wgs.local[0];
        }};
    }
};

class GroupNormalizationGeneratorCalcMeanVariance : public GroupNormalizationGeneratorBase {
public:
    GroupNormalizationGeneratorCalcMeanVariance() : GroupNormalizationGeneratorBase("group_normalization_fsv16", "calc_mean_variance") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GroupNormalizationGeneratorBase::get_jit_constants(params);
        jit.make("GROUP_NORM_KERNEL_GROUP_MEAN_VARIANCE", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const auto& ol = params.output_layouts[0];
            auto desc = params.typed_desc<group_normalization>();
            auto num_groups = static_cast<size_t>(desc->num_groups);
            auto f = extract_channel(ChannelName::FEATURE, ol);
            auto b = extract_channel(ChannelName::BATCH, ol);

            wgs.global[0] = f;
            wgs.global[1] = b;
            wgs.global[2] = 1;

            wgs.local[0] = f / num_groups;
            wgs.local[1] = 1;
            wgs.local[2] = 1;

            auto max_wgs = params.get_device_info().max_work_group_size;

            size_t divisor = 2;
            while (wgs.local[0] > max_wgs) {
                if ((f / num_groups) % divisor == 0) {
                    wgs.local[0] = (f / num_groups) / divisor;
                }
                divisor += 1;
            }
        }};
    }
};

class GroupNormalizationGeneratorFinalKernel : public GroupNormalizationGeneratorBase {
public:
    GroupNormalizationGeneratorFinalKernel() : GroupNormalizationGeneratorBase("group_normalization_fsv16", "final") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GroupNormalizationGeneratorBase::get_jit_constants(params);
        jit.make("GROUP_NORM_KERNEL_FINAL", 1);

        if (params.has_fused_primitives()) {
            const auto& out_l = params.get_output_layout(0);
            FusedOpsConfiguration conf = {"", std::vector<std::string>{"(b)", "(f)", "(y)", "(x)"}, "normalized", out_l.data_type};
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
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        add_fused_ops_arguments(args, params);
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const auto& ol = params.output_layouts[0];
            auto desc = params.typed_desc<group_normalization>();

            auto x = extract_channel(ChannelName::X, ol);
            auto y = extract_channel(ChannelName::Y, ol);
            auto f = extract_channel(ChannelName::FEATURE, ol);
            auto b = extract_channel(ChannelName::BATCH, ol);

            auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;

            wgs.global[0] = x * y;
            wgs.global[1] = ceil_div(f, fsv) * b;
            wgs.global[2] = 1;

            wgs.local[0] = x * y;
            wgs.local[1] = ceil_div(f, fsv) * b;
            wgs.local[2] = 1;

            size_t divisor = 2;
            while (wgs.local[0] > (max_wgs / fsv)) {
                if (wgs.global[0] % divisor == 0) {
                    wgs.local[0] = wgs.global[0] / divisor;
                }
                divisor += 1;
            }

            wgs.local[0] *= fsv;
            wgs.global[0] *= fsv;

            divisor = 2;
            while ((wgs.local[0] * wgs.local[1]) > max_wgs) {
                if (wgs.global[1] % divisor == 0) {
                    wgs.local[1] = wgs.global[1] / divisor;
                }
                divisor += 1;
            }
        }};
    }
};

class GroupNormalizationFsv16OptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GroupNormalizationFsv16OptImpl)

    Stage::Ptr calc_sqr_mean = make_stage<GroupNormalizationGeneratorCalcSQRMean>();
    Stage::Ptr calc_mean_variance = make_stage<GroupNormalizationGeneratorCalcMeanVariance>();
    Stage::Ptr final_normalize = make_stage<GroupNormalizationGeneratorFinalKernel>();

    GroupNormalizationFsv16OptImpl() : PrimitiveImplOCL(GroupNormalizationFsv16Opt::get_type_info_static()) {}
    GroupNormalizationFsv16OptImpl(const program_node& node, const RuntimeParams& params) : GroupNormalizationFsv16OptImpl() {
        add_stage(calc_sqr_mean, params);
        add_stage(calc_mean_variance, params);
        add_stage(final_normalize, params);
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GroupNormalizationFsv16OptImpl>(this);
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        auto desc = params.typed_desc<group_normalization>();
        const auto& shape = params.output_layouts[0].get_shape();
        auto buf = BufferDescriptor{shape[0] * align_to(shape[1], fsv), ov::element::f32};
        return {buf, buf};
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GroupNormalizationFsv16Opt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<group_normalization>());
    return std::make_unique<GroupNormalizationFsv16OptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GroupNormalizationFsv16OptImpl)
