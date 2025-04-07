// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "convolution_kernel_b_fs_yx_fsv16_depthwise.h"
#include "kernel_selector_utils.h"
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
    return k;
}

DeviceFeaturesKey ConvolutionKernel_b_fs_yx_fsv16_depthwise::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

bool ConvolutionKernel_b_fs_yx_fsv16_depthwise::Validate(const Params& p) const {
    const convolution_params& cp = static_cast<const convolution_params&>(p);

    if (cp.groups == 1)
        return false;

    if (cp.inputs[0].Feature().v != cp.groups || cp.outputs[0].Feature().v != cp.groups)
        return false;

    // Check that padding features doesn't miss-align the blocks
    if (cp.inputs[0].Feature().pad.before % feature_block_size != 0 || cp.outputs[0].Feature().pad.before % feature_block_size != 0)
        return false;

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
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.outputs[0].X().v, block_width)));
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", (params.outputs[0].X().v != 1) ? 8 : 1));
    jit.AddConstant(MakeJitConstant("IC_BLOCK", feature_block_size));
    jit.AddConstant(MakeJitConstant("FILTER_SIZE_X_DIV_2", params.filterSize.x / 2));
    if (params.outputs[0].Feature().v % feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
    }

    return jit;
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16_depthwise::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

}  // namespace kernel_selector
