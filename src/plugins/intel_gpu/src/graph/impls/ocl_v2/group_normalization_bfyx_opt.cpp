// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_bfyx_opt.hpp"

#include <string_view>

#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "ocl_v2/utils/fused_ops_jitter.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {

using namespace ov::intel_gpu::ocl;

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

JitConstants make_work_group_jit_constants(const DispatchDataFunc& func, const RuntimeParams& params) {
    JitConstants jit;
    if (params.is_dynamic()) {
        jit.add({
            make_jit_constant("GWS0", "get_global_size(0)"),
            make_jit_constant("LWS0", "get_local_size(0)"),
            make_jit_constant("LWS1", "get_local_size(1)"),
            make_jit_constant("LWS2", "get_local_size(2)"),
        });
    } else {
        KernelData kd;
        func(params, kd);
        jit.add({
            make_jit_constant("GWS0", kd.params.workGroups.global[0]),
            make_jit_constant("LWS0", kd.params.workGroups.local[0]),
            make_jit_constant("LWS1", kd.params.workGroups.local[1]),
            make_jit_constant("LWS2", kd.params.workGroups.local[2]),
        });
    }

    return jit;
}

class GroupNormalizationGeneratorBfyxOptBase : public KernelGenerator {
public:
    explicit GroupNormalizationGeneratorBfyxOptBase(std::string_view name, std::string_view suffix) : KernelGenerator(name, suffix) {}
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<group_normalization>();
        jit.make("EPSILON", static_cast<float>(desc->epsilon));
        jit.make("NUM_GROUPS", desc->num_groups);

        auto activation_type = get_activation_type(params);
        jit.add(make_type_jit_constants("ACTIVATION", activation_type));
        jit.add(make_type_jit_constants("ACCUMULATOR", get_accumulator_type(params)));

        return jit;
    }
};

class GroupNormalizationGeneratorCalcSQRMean : public GroupNormalizationGeneratorBfyxOptBase {
public:
    GroupNormalizationGeneratorCalcSQRMean() : GroupNormalizationGeneratorBfyxOptBase("group_normalization_bfyx_opt", "calc_sqr_mean") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = GroupNormalizationGeneratorBfyxOptBase::get_jit_constants(params);
        jit_constants.make("GROUP_NORM_KERNEL_FEATURE_MEAN_SQR_MEAN", 1);
        jit_constants.add(make_work_group_jit_constants(get_dispatch_data_func(), params));
        return jit_constants;
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
            auto z = extract_channel(ChannelName::Z, ol);
            auto f = extract_channel(ChannelName::FEATURE, ol);
            auto b = extract_channel(ChannelName::BATCH, ol);

            wgs.global[0] = x;
            wgs.global[1] = y;
            wgs.global[2] = z * f * b;

            wgs.local[0] = x;
            wgs.local[1] = y;
            wgs.local[2] = z;

            auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;

            if ((x * y * z) > max_wgs) {
                if (z > max_wgs) {
                    wgs.local[0] = 1;
                    wgs.local[1] = 1;
                    for (size_t lws = 2; lws <= z; ++lws) {
                        if (z % lws == 0 && (z / lws) <= max_wgs) {
                            wgs.local[2] = z / lws;
                            wgs.global[2] = wgs.local[2] * f * b;
                            break;
                        }
                    }
                } else {
                    if ((y * z) > max_wgs) {
                        wgs.local[0] = 1;
                        for (size_t lws = 2; lws <= y; ++lws) {
                            if (y % lws == 0 && (y / lws * z) <= max_wgs) {
                                wgs.local[1] = y / lws;
                                break;
                            }
                        }
                    } else {
                        for (size_t lws = 2; lws <= x; ++lws) {
                            if (x % lws == 0 && (x / lws * y * z) <= max_wgs) {
                                wgs.local[0] = x / lws;
                                break;
                            }
                        }
                    }
                }
            }
            wgs.global[0] = wgs.local[0];
            wgs.global[1] = wgs.local[1];
        }};
    }
};

class GroupNormalizationGeneratorCalcMeanVariance : public GroupNormalizationGeneratorBfyxOptBase {
public:
    GroupNormalizationGeneratorCalcMeanVariance() : GroupNormalizationGeneratorBfyxOptBase("group_normalization_bfyx_opt", "calc_mean_variance") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = GroupNormalizationGeneratorBfyxOptBase::get_jit_constants(params);
        jit_constants.make("GROUP_NORM_KERNEL_GROUP_MEAN_VARIANCE", 1);
        jit_constants.add(make_work_group_jit_constants(get_dispatch_data_func(), params));
        return jit_constants;
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

            auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;

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

class GroupNormalizationGeneratorFinalKernel : public GroupNormalizationGeneratorBfyxOptBase {
public:
    GroupNormalizationGeneratorFinalKernel() : GroupNormalizationGeneratorBfyxOptBase("group_normalization_bfyx_opt", "final") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GroupNormalizationGeneratorBfyxOptBase::get_jit_constants(params);
        jit.make("GROUP_NORM_KERNEL_FINAL", 1);
        jit.add(make_work_group_jit_constants(get_dispatch_data_func(), params));

        if (params.has_fused_primitives()) {
            const auto& out_l = params.get_output_layout(0);
            std::vector<std::string> idx_order;
            if (out_l.get_rank() == 5) {
                idx_order = {"(b)", "(f)", "(z)", "(y)", "(x)"};
            } else if (out_l.get_rank() <= 4) {
                idx_order = {"(b)", "(f)", "(y)", "(x)"};
            } else {
                OPENVINO_THROW("GroupNormalizationGeneratorBfyxOpt doesn't support rank ", out_l.get_rank());
            }

            FusedOpsConfiguration conf = {"", idx_order, "normalized", get_activation_type(params), 1};
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
            auto z = extract_channel(ChannelName::Z, ol);
            auto f = extract_channel(ChannelName::FEATURE, ol);
            auto b = extract_channel(ChannelName::BATCH, ol);

            auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;

            wgs.global[0] = x * y * z;
            wgs.global[1] = f * b;
            wgs.global[2] = 1;

            wgs.local[0] = x * y * z;
            wgs.local[1] = f * b;
            wgs.local[2] = 1;

            size_t divisor = 2;
            while (wgs.local[0] > max_wgs) {
                if (wgs.global[0] % divisor == 0) {
                    wgs.local[0] = wgs.global[0] / divisor;
                }
                divisor += 1;
            }

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

class GroupNormalizationBfyxOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GroupNormalizationBfyxOptImpl)
    Stage::Ptr calc_sqr_mean = make_stage<GroupNormalizationGeneratorCalcSQRMean>();
    Stage::Ptr calc_mean_variance = make_stage<GroupNormalizationGeneratorCalcMeanVariance>();
    Stage::Ptr normalize = make_stage<GroupNormalizationGeneratorFinalKernel>();

    GroupNormalizationBfyxOptImpl() : PrimitiveImplOCL(GroupNormalizationBfyxOpt::get_type_info_static()) {}
    GroupNormalizationBfyxOptImpl(const program_node& node, const RuntimeParams& params) : GroupNormalizationBfyxOptImpl() {
        add_stage(calc_sqr_mean, params);
        add_stage(calc_mean_variance, params);
        add_stage(normalize, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GroupNormalizationBfyxOptImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        auto desc = params.typed_desc<group_normalization>();
        const auto& shape = params.output_layouts[0].get_shape();
        auto buf = BufferDescriptor{shape[0] * shape[1], ov::element::f32};
        return {buf, buf};
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GroupNormalizationBfyxOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<group_normalization>());
    return std::make_unique<GroupNormalizationBfyxOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GroupNormalizationBfyxOptImpl)
