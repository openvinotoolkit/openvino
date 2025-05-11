// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_imad.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <vector>
#include <iostream>

//
// Kernel specific constants
//
#define SIMD_SIZE 16

static void getOutBlock_WH(size_t output_size,
                           size_t stride,
                           size_t kernel_size,
                           size_t dilation,
                           size_t& output_block_w,
                           size_t& output_block_h) {
    output_block_w = output_block_h = 0;

    size_t upper_border = output_size < SIMD_SIZE ? output_size : SIMD_SIZE;

    size_t stride_restrictions = (SIMD_SIZE - (kernel_size - 1) * dilation - 1) / stride + 1;

    size_t max_posible_tile_size = upper_border < stride_restrictions ? upper_border : stride_restrictions;

    if (output_size % max_posible_tile_size == 0) {
        output_block_w = max_posible_tile_size;
    } else {
        size_t min_horisontal_block_size = 2; // 4;

        size_t block_size = 0;

        for (size_t i = min_horisontal_block_size; i <= max_posible_tile_size; i++) {
            if (output_size % i == 0)
                block_size = i;
        }

        if (block_size != 0) {
            output_block_w = block_size;
        } else {
            output_block_w = max_posible_tile_size;
        }
    }

    if (output_block_w <= 4)
        output_block_h = output_block_w;
    else
        output_block_h = 1;
}

namespace kernel_selector {

ParamsKey ConvolutionKernel_imad::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::UINT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);

    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);

    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableGroupedConvolution();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_WEIGHTS);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS);
    return k;
}

DeviceFeaturesKey ConvolutionKernel_imad::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_blocked_read_write();
    k.requires_reqd_subgroup_size();
    k.requires_subgroups();
    k.requires_subgroup_broadcast();

    return k;
}

KernelsData ConvolutionKernel_imad::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

JitConstants ConvolutionKernel_imad::GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const {
    auto mem_consts = Parent::GetJitConstants(params, dispatchData);

    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];
    const auto& weights = params.weights;

    size_t in_fsv = 4;
    if (params.inputs[0].GetLayout() == DataLayout::b_fs_yx_fsv4)
        in_fsv = 4;
    else if (params.inputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16)
        in_fsv = 16;

    mem_consts.AddConstants({
        MakeJitConstant("_ID", RoundUp(input.Feature().v, in_fsv)),
        MakeJitConstant("IWPAD", input.X().pad.Total()),
        MakeJitConstant("IHPAD", input.Y().pad.Total()),
        MakeJitConstant("_OD", Align(output.Feature().v, SIMD_SIZE)),
        MakeJitConstant("OWPAD", output.X().pad.Total()),
        MakeJitConstant("OHPAD", output.Y().pad.Total()),
        MakeJitConstant("SIMD_SIZE", SIMD_SIZE),
        MakeJitConstant("FSV", in_fsv),
    });

    if (params.filterSize.x != 3 || params.filterSize.y != 3) {
        mem_consts.AddConstants({MakeJitConstant("NON_BLOCK_LOAD", 1)});
    }

    mem_consts.Merge(MakeTypeJitConstants(GetPackedInputType(params), "PACKED"));

    size_t obw, obh;
    getOutBlock_WH(output.X().v, params.stride.x, weights.X().v, params.dilation.x, obw, obh);
    mem_consts.AddConstants({MakeJitConstant("OUT_BLOCK_WIDTH", obw),
                             MakeJitConstant("OUT_BLOCK_HEIGHT", obh)});

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_scalar = {"", {"batch", "f", "(or+r)", "(oc+c)"}, "res", input_dt, 1 };
        conf_scalar.SetLoopAxes({Tensor::DataChannelName::Y, Tensor::DataChannelName::X});
        mem_consts.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return mem_consts;
}  // GetJitConstants

ConvolutionKernelBase::DispatchData ConvolutionKernel_imad::SetDefault(const convolution_params& params,
                                                                       int) const {
    DispatchData dispatchData;

    const auto& output = params.outputs[0];
    const auto& weights = params.weights;

    size_t otw, oth;
    getOutBlock_WH(output.X().v, params.stride.x, weights.X().v, params.dilation.x, otw, oth);

    dispatchData.gws = { // number of tiles needed to cover output width
                         CeilDiv(output.X().v, otw),

                         // number of tiles needed to cover output height
                         CeilDiv(output.Y().v, oth),

                         // round depth range up
                         Align(weights.OFM().v, SIMD_SIZE) * params.groups * output.Batch().v };

    dispatchData.lws = {1, 1, SIMD_SIZE};

    dispatchData.cldnnStyle = {0, 0, 0, 0, 0};
    dispatchData.gemmStyle = {0, 0, 0, 0, 0, 0};

    return dispatchData;
}  // SetDefault

KernelsPriority ConvolutionKernel_imad::GetKernelsPriority(const Params& /*params*/) const {
    // This kernel is quite slow for 1x1 and KHx1 kernels
    // TODO: check if we need any optimized kernels in this layout
    // If yes, we need to implement some customization for these cases.
    return FORCE_PRIORITY_3;
}

bool ConvolutionKernel_imad::Validate(const Params& params) const {
    if (!Parent::Validate(params)) {
        return false;
    }

    auto& conv_params = static_cast<const convolution_params&>(params);
    if (conv_params.groups > 1 && conv_params.weights.IFM().v % 4 != 0 &&
        conv_params.inputs[0].GetLayout() != DataLayout::b_fs_yx_fsv16)
        return false;

    size_t min_block_size_x = (conv_params.weights.X().v - 1) * conv_params.dilation.x + 1;
    if (min_block_size_x > SIMD_SIZE)
        return false;

    if (conv_params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS) {
        if ((conv_params.activations_zero_points.empty() || conv_params.weights_zero_points.empty()) &&
            (conv_params.compensation.empty()))
            return false;
    } else if (conv_params.quantization == QuantizationType::ASYMMETRIC_DATA) {
        if ((conv_params.activations_zero_points.empty()) &&
            (conv_params.compensation.empty()))
            return false;
    } else if (conv_params.quantization == QuantizationType::ASYMMETRIC_WEIGHTS) {
        if (conv_params.weights_zero_points.empty())
            return false;
    } else {
        if (!conv_params.activations_zero_points.empty() ||
            !conv_params.weights_zero_points.empty() ||
            !conv_params.compensation.empty())
            return false;
    }

    return true;
}
}  // namespace kernel_selector
