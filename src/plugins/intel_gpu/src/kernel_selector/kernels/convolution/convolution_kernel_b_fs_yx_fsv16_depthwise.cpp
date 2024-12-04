﻿// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "convolution_kernel_b_fs_yx_fsv16_depthwise.h"
#include "kernel_selector_utils.h"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include <string>

namespace kernel_selector {
static const size_t sub_group_size = 16;
static const size_t feature_block_size = 16;

ParamsKey ConvolutionKernel_b_fs_yx_fsv16_depthwise::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F16);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);

    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableGroupedConvolution();
    k.EnableDilation();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_b_fs_yx_fsv16_depthwise::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

bool ConvolutionKernel_b_fs_yx_fsv16_depthwise::Validate(const Params& p) const {
    const convolution_params& cp = static_cast<const convolution_params&>(p);

    if (cp.groups == 1) {
        GPU_DEBUG_LOG << " cp.groups == 1" << std::endl;
        return false;
    }

    // Check that features and groups are different
    // Check that padding features doesn't miss-align the blocks
    auto& input_feature = cp.inputs[0].Feature();
    if (!input_feature.is_dynamic) {
        if (input_feature.v != cp.groups || input_feature.pad.before % feature_block_size != 0) {
            GPU_DEBUG_LOG << " input_feature(" << input_feature.v << ") != cp.groups(" << cp.groups
                          << ") || input_feature.pad.before(" << input_feature.pad.before << ") != feature_block_size(" << feature_block_size << ")" << std::endl;
            return false;
        }
    }
    auto& output_feature = cp.outputs[0].Feature();
    if (!output_feature.is_dynamic) {
        if (output_feature.v != cp.groups || output_feature.pad.before % feature_block_size != 0) {
            GPU_DEBUG_LOG << " output_feature(" << output_feature.v << ") != cp.groups(" << cp.groups
                          << ") || output_feature.pad.before(" << output_feature.pad.before << ") != feature_block_size(" << feature_block_size << ")" << std::endl;
            return false;
        }
    }

    return true;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_b_fs_yx_fsv16_depthwise::SetDefault(const convolution_params& params,
                                                                                          int) const {
    DispatchData dispatchData = Parent::SetDefault(params);
    const auto& out = params.outputs[0];

    size_t x_block_size = (out.X().v != 1) ? 8 : 1;
    dispatchData.gws[0] = CeilDiv(out.X().v, x_block_size) * out.Y().v;
    dispatchData.gws[1] = Align(out.Feature().v, feature_block_size);
    dispatchData.gws[2] = out.Batch().v;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = sub_group_size;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_b_fs_yx_fsv16_depthwise::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const convolution_params&>(params);

    return p.outputs[0].Batch().v == 1 ? FORCE_PRIORITY_1 : FORCE_PRIORITY_7;
}

JitConstants ConvolutionKernel_b_fs_yx_fsv16_depthwise::GetJitConstants(const convolution_params& params,
                                                                        const DispatchData& dispatchData) const {
    auto jit = ConvolutionKernelBase::GetJitConstants(params, dispatchData);

    const size_t block_width = 8;

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_vec = { "_VEC", {"b", "(f_block*16)", "y", "x"},
                                           "dst",
                                           input_dt,
                                           block_width,
                                           LoadType::LT_ALIGNED_READ,
                                           BoundaryCheck::ENABLED,
                                           IndexType::TENSOR_COORD,
                                           Tensor::DataChannelName::X };
        FusedOpsConfiguration conf_scalar = { "_SCALAR",
                                              {"b", "(f_block*16)", "y", "(x+i)"},
                                              "dst[i]",
                                              input_dt,
                                              1,
                                              LoadType::LT_ALIGNED_READ,
                                              BoundaryCheck::ENABLED,
                                              IndexType::TENSOR_COORD,
                                              Tensor::DataChannelName::X };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_vec, conf_scalar}));
    }

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("IC_BLOCK", feature_block_size));
    jit.AddConstant(MakeJitConstant("FILTER_SIZE_X_DIV_2", params.filterSize.x / 2));

    std::cout << "params.has_dynamic_inputs(): " << params.has_dynamic_inputs() << std::endl;
    std::cout << "params.has_dynamic_outputs(): " << params.has_dynamic_outputs() << std::endl;

    if (params.has_dynamic_outputs()) {
        DimensionAccessHelperJit output_dims(params.outputs[0]);
        const auto block_width_str = std::to_string(block_width);
        const auto x_blocks = "(" + output_dims.x() + "+" + block_width_str + " - 1) / " + block_width_str;
        jit.AddConstant(MakeJitConstant("X_BLOCKS", x_blocks));

        const auto x_block_size = "((" + output_dims.x() + " != 1) ? 8 : 1)";
        jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", x_block_size));

        const auto output_leftover_num = "(" + output_dims.f() + "%" + std::to_string(feature_block_size) + ")";
        const auto output_leftover = "(" + output_leftover_num + "!= 0)";
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", output_leftover));
    } else {
        jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.outputs[0].X().v, block_width)));
        jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", (params.outputs[0].X().v != 1) ? 8 : 1));
        if (params.outputs[0].Feature().v % feature_block_size != 0) {
            jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
        }
    }

    return jit;
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16_depthwise::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

}  // namespace kernel_selector
