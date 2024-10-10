// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_bfyx_opt.hpp"
#include "impls/ocl/jitter.hpp"
#include "impls/ocl/kernel_base.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "openvino/core/partial_shape.hpp"
#include "primitive_ocl_base.hpp"

namespace cldnn {
namespace ocl {

namespace {

using namespace ov::intel_gpu::ocl;

ov::element::Type get_activation_type(const program_node& node) {
    if (node.get_input_layout(0).data_type == ov::element::f16)
        return ov::element::f16;
    return ov::element::f32;
}

ov::element::Type get_accumulator_type(const program_node& node) {
    auto in_dt = node.get_input_layout(0).data_type;
    switch (in_dt) {
    case ov::element::u8:
    case ov::element::i8:
        return ov::element::i32;
    default:
        return ov::element::f32;
    }
}

DispatchData get_stage1_dispatch_data(const kernel_impl_params& params) {
    WorkGroupSizes dispatch_data;

    if (!params.is_dynamic()) {
        const auto& ol = params.output_layouts[0];
        auto x = extract_channel(ChannelName::X, ol);
        auto y = extract_channel(ChannelName::Y, ol);
        auto z = extract_channel(ChannelName::Z, ol);
        auto f = extract_channel(ChannelName::FEATURE, ol);
        auto b = extract_channel(ChannelName::BATCH, ol);

        dispatch_data.global[0] = x;
        dispatch_data.global[1] = y;
        dispatch_data.global[2] = z * f * b;

        dispatch_data.local[0] = x;
        dispatch_data.local[1] = y;
        dispatch_data.local[2] = z;

        auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;

        if ((x * y * z) > max_wgs) {
            if (z > max_wgs) {
                dispatch_data.local[0] = 1;
                dispatch_data.local[1] = 1;
                for (size_t lws = 2; lws <= z; ++lws) {
                    if (z % lws == 0 && (z / lws) <= max_wgs) {
                        dispatch_data.local[2] = z / lws;
                        dispatch_data.global[2] = dispatch_data.local[2] * f * b;
                        break;
                    }
                }
            } else {
                if ((y * z) > max_wgs) {
                    dispatch_data.local[0] = 1;
                    for (size_t lws = 2; lws <= y; ++lws) {
                        if (y % lws == 0 && (y / lws * z) <= max_wgs) {
                            dispatch_data.local[1] = y / lws;
                            break;
                        }
                    }
                } else {
                    for (size_t lws = 2; lws <= x; ++lws) {
                        if (x % lws == 0 && (x / lws * y * z) <= max_wgs) {
                            dispatch_data.local[0] = x / lws;
                            break;
                        }
                    }
                }
            }
        }
    }

    return { dispatch_data, {} };
}

DispatchData get_stage2_dispatch_data(const kernel_impl_params& params) {
    WorkGroupSizes dispatch_data;
    if (!params.is_dynamic()) {
        const auto& ol = params.output_layouts[0];
        auto desc = params.typed_desc<group_normalization>();
        size_t num_groups = static_cast<std::size_t>(desc->num_groups);
        auto f = extract_channel(ChannelName::FEATURE, ol);
        auto b = extract_channel(ChannelName::BATCH, ol);

        dispatch_data.global[0] = f;
        dispatch_data.global[1] = b;
        dispatch_data.global[2] = 1;

        dispatch_data.local[0] = f / num_groups;
        dispatch_data.local[1] = 1;
        dispatch_data.local[2] = 1;

        auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;

        size_t divisor = 2;
        while (dispatch_data.local[0] > max_wgs) {
            if ((f / num_groups) % divisor == 0) {
                dispatch_data.local[0] = (f / num_groups) / divisor;
            }
            divisor += 1;
        }
    }

    return { dispatch_data, {} };
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

class CalcFeatureMeanKernel : public SingleKernelGenerator {
public:
    CalcFeatureMeanKernel() : SingleKernelGenerator("group_normalization_gpu_bfyx_opt") {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit_constants = SingleKernelGenerator::get_jit_constants(node, params);
        jit_constants.make("GROUP_NORM_KERNEL_FEATURE_MEAN", 1);
        jit_constants.add(make_work_group_jit_constants(get_dispatch_data(params).work_groups, params.is_dynamic()));
        return jit_constants;
    }

    Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

        return args;
    }

    DispatchDataFunc get_dispatch_data_func(const kernel_impl_params& params) const override {
        return get_stage1_dispatch_data;
    }

    std::vector<layout> get_interanl_buffers(const program_node& node, const kernel_impl_params& params) const override {
        const auto& pshape = params.output_layouts[0].get_partial_shape();
        if (pshape.is_static()) {
            int64_t batch = pshape[0].get_length();
            int64_t feature = pshape[1].get_length();
            auto buf_layout = layout{ov::PartialShape{batch * feature}, ov::element::f32, format::bfyx };
            return { buf_layout, buf_layout };
        } else {
            auto buf_layout = layout{ov::PartialShape{1}, ov::element::f32, format::bfyx };
            return { buf_layout, buf_layout };
        }
    }
};

class CalcGroupMeanKernel : public SingleKernelGenerator {
public:
    CalcGroupMeanKernel() : SingleKernelGenerator("group_normalization_gpu_bfyx_opt") {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit_constants = SingleKernelGenerator::get_jit_constants(node, params);
        jit_constants.make("GROUP_NORM_KERNEL_GROUP_MEAN", 1);
        jit_constants.add(make_work_group_jit_constants(get_dispatch_data(params).work_groups, params.is_dynamic()));
        return jit_constants;
    }

    Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        return args;
    }

    DispatchDataFunc get_dispatch_data_func(const kernel_impl_params& params) const override {
        return get_stage2_dispatch_data;
    }
};

class CalcFeatureVarKernel : public SingleKernelGenerator {
public:
    CalcFeatureVarKernel() : SingleKernelGenerator("group_normalization_gpu_bfyx_opt") {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit_constants = SingleKernelGenerator::get_jit_constants(node, params);
        jit_constants.make("GROUP_NORM_KERNEL_FEATURE_VAR", 1);
        jit_constants.add(make_work_group_jit_constants(get_dispatch_data(params).work_groups, params.is_dynamic()));
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
        return get_stage1_dispatch_data;
    }
};

class CalcGroupVarKernel : public SingleKernelGenerator {
public:
    CalcGroupVarKernel() : SingleKernelGenerator("group_normalization_gpu_bfyx_opt") {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit_constants = SingleKernelGenerator::get_jit_constants(node, params);
        jit_constants.make("GROUP_NORM_KERNEL_GROUP_VAR", 1);
        jit_constants.add(make_work_group_jit_constants(get_dispatch_data(params).work_groups, params.is_dynamic()));
        return jit_constants;
    }

    Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        return args;
    }

    DispatchDataFunc get_dispatch_data_func(const kernel_impl_params& params) const override {
        return get_stage2_dispatch_data;
    }
};

class FinalKernel : public SingleKernelGenerator {
public:
    FinalKernel() : SingleKernelGenerator("group_normalization_gpu_bfyx_opt") {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit_constants = SingleKernelGenerator::get_jit_constants(node, params);
        jit_constants.make("GROUP_NORM_KERNEL_FINAL", 1);
        jit_constants.add(make_work_group_jit_constants(get_dispatch_data(params).work_groups, params.is_dynamic()));
        return jit_constants;
    }

    Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});

        return args;
    }

    DispatchDataFunc get_dispatch_data_func(const kernel_impl_params& params) const override {
        static auto f = DISPATCH_DATA_FUNC(params) {
            WorkGroupSizes dispatch_data;

            if (!params.is_dynamic()) {
                const auto& ol = params.output_layouts[0];
                auto desc = params.typed_desc<group_normalization>();

                auto x = extract_channel(ChannelName::X, ol);
                auto y = extract_channel(ChannelName::Y, ol);
                auto z = extract_channel(ChannelName::Z, ol);
                auto f = extract_channel(ChannelName::FEATURE, ol);
                auto b = extract_channel(ChannelName::BATCH, ol);

                auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;

                dispatch_data.global[0] = x * y * z;
                dispatch_data.global[1] = f * b;
                dispatch_data.global[2] = 1;

                dispatch_data.local[0] = x * y * z;
                dispatch_data.local[1] = f * b;
                dispatch_data.local[2] = 1;

                size_t divisor = 2;
                while (dispatch_data.local[0] > max_wgs) {
                    if (dispatch_data.global[0] % divisor == 0) {
                        dispatch_data.local[0] = dispatch_data.global[0] / divisor;
                    }
                    divisor += 1;
                }

                divisor = 2;
                while ((dispatch_data.local[0] * dispatch_data.local[1]) > max_wgs) {
                    if (dispatch_data.global[1] % divisor == 0) {
                        dispatch_data.local[1] = dispatch_data.global[1] / divisor;
                    }
                    divisor += 1;
                }
            }

            return { dispatch_data, {} };
        };
        return f;
    }
};

class GroupNormalizationGeneratorBfyxOpt : public MultiStageKernelGenerator {
public:
    GroupNormalizationGeneratorBfyxOpt() : MultiStageKernelGenerator(
            CalcFeatureMeanKernel(),
            CalcGroupMeanKernel(),
            CalcFeatureVarKernel(),
            CalcGroupVarKernel(),
            FinalKernel()
    ) {}

    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        JitConstants jit;
        auto desc = params.typed_desc<group_normalization>();
        jit.make("EPSILON", static_cast<float>(desc->epsilon));
        jit.make("NUM_GROUPS", desc->num_groups);

        auto activation_type = get_activation_type(node);
        jit.add(make_type_jit_constants("ACTIVATION", activation_type));
        jit.add(make_type_jit_constants("ACCUMULATOR", get_accumulator_type(node)));

        // if (!params.fused_ops.empty()) {
        //     std::vector<std::string> idx_order;
        //     if (params.inputs[0].GetDims().size() == 5) {
        //         idx_order = { "(b)", "(f)", "(z)", "(y)", "(x)" };
        //     } else if (params.inputs[0].GetDims().size() <= 4) {
        //         idx_order = { "(b)", "(f)", "(y)", "(x)" };
        //     } else {
        //         OPENVINO_THROW("group_normalization_bfyx doesn't support 5D or higher dims.");
        //     }
        //     auto conf = FusedOpsConfiguration("", idx_order, "normalized", activation_type, 1);
        //     jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
        // }

        return jit;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GroupNormalizationBfyxOpt::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<group_normalization>());
    GroupNormalizationGeneratorBfyxOpt gen;
    return cldnn::make_unique<primitive_impl_ocl>(gen.get_kernels_data(node, params), std::string(get_type_info().name));
}

}  // namespace ocl
}  // namespace cldnn
