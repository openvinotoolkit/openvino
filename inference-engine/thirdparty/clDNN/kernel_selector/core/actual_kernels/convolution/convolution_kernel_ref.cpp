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
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();

    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfzyx_f16);
    k.EnableOutputLayout(DataLayout::bfzyx_f16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableDepthwiseSeparableOpt();
    k.EnableInt8Quantization();
    k.EnableOutputCalibration();
    k.DisableTuning();
    k.EnableLocalConvolution();
    k.EnableGroupedConvolution();
    return k;
}

KernelsData ConvolutionKernel_Ref::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
JitConstants ConvolutionKernel_Ref::GetJitConstants(const convolution_params& params, const DispatchData& kd) const {
    JitConstants jit = ConvolutionKernelBase::GetJitConstants(params, kd);

    // Create an ACTIVATION macro accepting type parameter - we don't have a
    // single UNIT_TYPE for the whole kernel.
    //
    // TODO: This gives both ACTIVATION and ACTIVATION_TYPED. Should we
    // factor that out into a virtual function to avoid creation of similar
    // yet distinct macros?
    jit.Merge(MakeActivationJitConstants(params.activations, "_CONV_TYPED", true));
    // Needs to be done on host to get _MAX_VAL/_MIN_VAL/TO_TYPE macros
    // available (will be used in the activation).
    //
    // TODO: Should it be done for all the kernels? Might even be done
    // directly in the OpenCL include, as opposite to jitting. On the other
    // hand, going through jit ensures we are in sync with the
    // MakeTypeJitConstants implementation.
    jit.Merge(MakeTypeJitConstants(Datatype::F32, "float"));

    if (params.int8_quantization && !params.bias.empty() && params.bias[0].GetDType() == Datatype::F32)
        jit.AddConstant(MakeJitConstant("DONT_DEQUANTIZE_BIAS", "1"));

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

    auto local = GetOptimalLocalWorkGroupSizes(global);

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

    // The only supported u8 input is the one with quantization, would
    // require some additional checks.

    if (input_type == output_type && input_type != Datatype::UINT8)
        return true;

    // Otherwise, only i8/u8 -> i8/u8/fp32 convolution with i8 weights and i32 biases
    // with quantization term is supported by now.
    if ((input_type != Datatype::INT8 && input_type != Datatype::UINT8) ||
        (output_type != Datatype::INT8 && output_type != Datatype::UINT8 && output_type != Datatype::F32))
        return false;

    if (!conv_params.int8_quantization)
        return false;

    if (conv_params.output_calibration)
        // Probably everything is in place to support the case, just need to add a test.
        return false;

    if (conv_params.weights.GetDType() != WeightsType::INT8)
        return false;

    if (!conv_params.bias.empty() && conv_params.bias.front().GetDType() != Datatype::INT32)
        // Non-quantized (FP32) bias is probably OK too, need to verify.
        return false;

    return true;
}
}  // namespace kernel_selector
