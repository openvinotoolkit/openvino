// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <string>

namespace kernel_selector {

ParamsKey ConvolutionKernel_Ref::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();

    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableGroupedConvolution();

    k.EnableDynamicShapesSupport();

    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_WEIGHTS);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS);
    return k;
}

KernelsData ConvolutionKernel_Ref::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}

JitConstants ConvolutionKernel_Ref::GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const {
    JitConstants jit = ConvolutionKernelBase::GetJitConstants(params, dispatchData);

    Datatype accumulator_dt;
    Datatype activation_dt;
    if (params.quantization != QuantizationType::NONE) {
        accumulator_dt = Datatype::INT32;
        activation_dt = Datatype::F32;
    } else {
        accumulator_dt = GetAccumulatorType(params);
        activation_dt = GetActivationType(params);
    }

    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(accumulator_dt, "ACCUMULATOR"));
    jit.Merge(MakeActivationJitConstants(params.activations, activation_dt, "_TYPED"));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 4) {
            idx_order = {"b", "f", "y", "x"};
        } else if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5) {
            idx_order = {"b", "f", "z", "y", "x"};
        }

        FusedOpsConfiguration conf = {"", idx_order, "dequantized", activation_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    return jit;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_Ref::SetDefault(const convolution_params& params,
                                                                      int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(params, autoTuneIndex);

    // FIXME: ConvolutionKernelBase::SetDefault should probably be pure and
    // not setting these at all as it's something specific to a concrete
    // implementation. Unfortunately, convolution classes are currently
    // written in such a way that most of the logic is in the base class'
    // method guarded by the "if" conditions (based on the layout!).
    //
    // Just set the correct value for a particular implementation here,
    // until the whole hierarchy is re-written.
    const auto& out = params.outputs[0];
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{Tensor::DataChannelName::X},
                                                                     {Tensor::DataChannelName::Y, Tensor::DataChannelName::Z},
                                                                     {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};

    dispatchData.gws = {out.X().v, out.Y().v * out.Z().v, out.Feature().v * out.Batch().v};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
    return dispatchData;
}

KernelsPriority ConvolutionKernel_Ref::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool ConvolutionKernel_Ref::Validate(const Params& params) const {
    if (!ConvolutionKernelBase::Validate(params))
        return false;

    const auto& conv_params = static_cast<const convolution_params&>(params);
    auto input_type = conv_params.inputs[0].GetDType();
    auto output_type = conv_params.outputs[0].GetDType();
    auto weights_type = conv_params.weights.GetDType();

    // int8/uint8 inputs (quantization case) require additional checks
    // require some additional checks.
    if (input_type != Datatype::UINT8 && input_type != Datatype::INT8 &&
        output_type != Datatype::UINT8 && output_type != Datatype::INT8)
        return true;

    // (u)int8 input + fp weights
    if (weights_type == WeightsType::F32 || weights_type == WeightsType::F16)
        return true;

    bool is_quantization = (input_type == Datatype::INT8 || input_type == Datatype::UINT8) &&
                           (output_type == Datatype::INT8 || output_type == Datatype::UINT8 ||
                            output_type == Datatype::F32 || output_type == Datatype::F16) &&
                           (weights_type == WeightsType::INT8);

    bool has_fused_op = (input_type == Datatype::F32 || input_type == Datatype::F16) &&
                        !conv_params.fused_ops.empty() &&
                        (output_type == Datatype::INT8 || output_type == Datatype::UINT8);

    if (!is_quantization && !has_fused_op)
        return false;

    if (conv_params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS) {
        if (conv_params.activations_zero_points.empty() || conv_params.weights_zero_points.empty())
            return false;
    } else if (conv_params.quantization == QuantizationType::ASYMMETRIC_DATA) {
        if (conv_params.activations_zero_points.empty())
            return false;
    } else if (conv_params.quantization == QuantizationType::ASYMMETRIC_WEIGHTS) {
        if (conv_params.weights_zero_points.empty())
            return false;
    } else {
        if (!conv_params.activations_zero_points.empty() || !conv_params.weights_zero_points.empty())
            return false;
    }

    return true;
}

}  // namespace kernel_selector
