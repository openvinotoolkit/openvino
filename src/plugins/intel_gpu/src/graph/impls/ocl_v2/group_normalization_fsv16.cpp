// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_fsv16.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "utils/jitter.hpp"
#include "utils/kernel_base.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "openvino/core/partial_shape.hpp"
#include "primitive_ocl_base.hpp"

namespace ov::intel_gpu::ocl {

namespace {

static constexpr size_t fsv = 16;
static constexpr size_t simd = fsv;

ov::element::Type get_activation_type(const kernel_impl_params& params) {
    if (params.get_input_layout(0).data_type == ov::element::f16)
        return ov::element::f16;
    return ov::element::f32;
}

ov::element::Type get_accumulator_type(const kernel_impl_params& params) {
    auto in_dt = params.get_input_layout(0).data_type;
    switch (in_dt) {
    case ov::element::u8:
    case ov::element::i8:
        return ov::element::i32;
    default:
        return ov::element::f32;
    }
}

JitConstants make_work_group_jit_constants(const WorkGroupSizes& wgs, bool is_dynamic) {
    JitConstants jit;
    if (is_dynamic) {
        jit.add({
            make_jit_constant("GWS0", "get_global_size(0)"),
            make_jit_constant("LWS0", "get_local_size(0)"),
            make_jit_constant("LWS1", "get_local_size(1)"),
            make_jit_constant("LWS2", "get_local_size(2)"),
        });
    } else {
        jit.add({
            make_jit_constant("GWS0", wgs.global[0]),
            make_jit_constant("LWS0", wgs.local[0]),
            make_jit_constant("LWS1", wgs.local[1]),
            make_jit_constant("LWS2", wgs.local[2]),
        });
    }

    return jit;
}

class GroupNormalizationGeneratorBfyxOptBase : public SingleKernelGenerator {
public:
    explicit GroupNormalizationGeneratorBfyxOptBase(std::string_view name) : SingleKernelGenerator(name) {}
    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = SingleKernelGenerator::get_jit_constants(params);
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

        // TODO: Handle fused ops

        return jit;
    }
};

class GroupNormalizationGeneratorCalcSQRMean : public GroupNormalizationGeneratorBfyxOptBase {
public:
    GroupNormalizationGeneratorCalcSQRMean() : GroupNormalizationGeneratorBfyxOptBase("group_normalization_fsv16") {}

protected:
    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit_constants = GroupNormalizationGeneratorBfyxOptBase::get_jit_constants(params);
        jit_constants.make("GROUP_NORM_KERNEL_FEATURE_MEAN_SQR_MEAN", 1);
        KernelData kd;
        get_dispatch_data_func()(params, kd);
        jit_constants.add(make_work_group_jit_constants(kd.params.workGroups, params.is_dynamic()));
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
        static auto f = DISPATCH_DATA_FUNC(params, kd, rt_params) {
            auto& wgs = kd.params.workGroups;

            if (!params.is_dynamic()) {
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
            }
        };

        return f;
    }
};

class GroupNormalizationGeneratorCalcMeanVariance : public GroupNormalizationGeneratorBfyxOptBase {
public:
    GroupNormalizationGeneratorCalcMeanVariance() : GroupNormalizationGeneratorBfyxOptBase("group_normalization_bfyx_opt") {}

protected:
    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit_constants = GroupNormalizationGeneratorBfyxOptBase::get_jit_constants(params);
        jit_constants.make("GROUP_NORM_KERNEL_GROUP_MEAN_VARIANCE", 1);
        KernelData kd;
        get_dispatch_data_func()(params, kd);
        jit_constants.add(make_work_group_jit_constants(kd.params.workGroups, params.is_dynamic()));
        return jit_constants;
    }

    Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        return args;
    }

    DispatchDataFunc get_dispatch_data_func() const override {
        static auto f = DISPATCH_DATA_FUNC(params, kd, rt_params) {
            auto& wgs = kd.params.workGroups;
            if (!params.is_dynamic()) {
                const auto& ol = params.output_layouts[0];
                auto desc = params.typed_desc<group_normalization>();
                size_t num_groups = static_cast<std::size_t>(desc->num_groups);
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
            }
        };

        return f;
    }
};

class GroupNormalizationGeneratorFinalKernel : public GroupNormalizationGeneratorBfyxOptBase {
public:
    GroupNormalizationGeneratorFinalKernel() : GroupNormalizationGeneratorBfyxOptBase("group_normalization_bfyx_opt") {}

protected:
    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit_constants = GroupNormalizationGeneratorBfyxOptBase::get_jit_constants(params);
        jit_constants.make("GROUP_NORM_KERNEL_FINAL", 1);
        KernelData kd;
        get_dispatch_data_func()(params, kd);
        jit_constants.add(make_work_group_jit_constants(kd.params.workGroups, params.is_dynamic()));
        return jit_constants;
    }

    Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});

        return args;
    }

    DispatchDataFunc get_dispatch_data_func() const override {
        static auto f = DISPATCH_DATA_FUNC(params, kd, rt_params) {
            auto& wgs = kd.params.workGroups;

            if (!params.is_dynamic()) {
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

                wgs.local[0] = x * y ;
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
            }
        };
        return f;
    }
};

class GroupNormalizationFsv16OptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GroupNormalizationFsv16OptImpl)

    Stage calc_sqr_mean = make_stage<GroupNormalizationGeneratorCalcSQRMean>();
    Stage calc_mean_variance = make_stage<GroupNormalizationGeneratorCalcMeanVariance>();
    Stage final_normalize = make_stage<GroupNormalizationGeneratorFinalKernel>();

    GroupNormalizationFsv16OptImpl() : PrimitiveImplOCL(GroupNormalizationFsv16Opt::get_type_info_static()) {}
    GroupNormalizationFsv16OptImpl(const program_node& node, const kernel_impl_params& params) : GroupNormalizationFsv16OptImpl() {
        add_stage(calc_sqr_mean, params);
        add_stage(calc_mean_variance, params);
        add_stage(final_normalize, params);
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GroupNormalizationFsv16OptImpl>(this);
    }

    std::vector<layout> get_internal_buffer_layouts(const kernel_impl_params& params) const override {
        auto desc = params.typed_desc<group_normalization>();
        const auto& shape = params.output_layouts[0].get_shape();
        auto buf = layout{ov::PartialShape{static_cast<int64_t>(shape[0] * align_to(shape[1], fsv))}, ov::element::f32, format::bfyx };
        return { buf, buf };
    }
};


}  // namespace

std::unique_ptr<primitive_impl> GroupNormalizationFsv16Opt::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<group_normalization>());
    return std::make_unique<GroupNormalizationFsv16OptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GroupNormalizationFsv16OptImpl)
