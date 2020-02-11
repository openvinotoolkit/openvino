/*
// Copyright (c) 2016-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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
    k.EnableSplitSupport();
    k.EnableDepthwiseSeparableOpt();
    k.DisableTuning();
    k.EnableLocalConvolution();
    k.EnableGroupedConvolution();

    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_WEIGHTS);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS);
    return k;
}

KernelsData ConvolutionKernel_Ref::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

JitConstants ConvolutionKernel_Ref::GetJitConstants(const convolution_params& params, const DispatchData& kd) const {
    JitConstants jit = ConvolutionKernelBase::GetJitConstants(params, kd);

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
        if (DataTensor::ChannelsCount(params.output.GetLayout()) == 4) {
            idx_order = {"b", "of", "y", "x"};
        } else if (DataTensor::ChannelsCount(params.output.GetLayout()) == 5) {
            idx_order = {"b", "of", "z", "y", "x"};
        }

        FusedOpsConfiguration conf = {"", idx_order, "dequantized", activation_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    return jit;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_Ref::SetDefault(const convolution_params& params,
                                                                      int autoTuneIndex) const {
    DispatchData kd = ConvolutionKernelBase::SetDefault(params, autoTuneIndex);

    // FIXME: ConvolutionKernelBase::SetDefault should probably be pure and
    // not setting these at all as it's something specific to a concrete
    // implementation. Unfortunately, convolution classes are currently
    // written in such a way that most of the logic is in the base class'
    // method guarded by the "if" conditions (based on the layout!).
    //
    // Just set the correct value for a particular implementation here,
    // until the whole hierarchy is re-written.
    const auto& out = params.output;
    std::vector<size_t> global = {out.X().v, out.Y().v * out.Z().v, out.Feature().v * out.Batch().v};

    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];
    return kd;
}

bool ConvolutionKernel_Ref::Validate(const Params& params, const optional_params& options) const {
    if (!ConvolutionKernelBase::Validate(params, options))
        return false;

    const auto& conv_params = static_cast<const convolution_params&>(params);
    auto input_type = conv_params.inputs[0].GetDType();
    auto output_type = conv_params.output.GetDType();
    auto weights_type = conv_params.weights.GetDType();

    // int8/uint8 inputs (quantization case) require additional checks
    // require some additional checks.
    if (input_type == output_type && input_type != Datatype::UINT8 && input_type != Datatype::INT8)
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
