// Copyright (c) 2016-2020 Intel Corporation
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


#include "deconvolution_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>
#include <algorithm>

namespace kernel_selector {
std::string deconvolution_params::to_string() const {
    std::stringstream s;

    s << base_params::to_string() << "_";
    if (bias.empty()) {
        s << "no_bias"
          << "_";
    } else {
        s << "bias_size:" << bias[0].PhysicalSize() << "_";
    }
    s << filterSize.x << "_" << filterSize.y << "_";
    s << stride.x << "_" << stride.y << "_";
    s << dilation.x << "_" << dilation.y << "_";
    s << padding.x << "_" << padding.y << "_";
    s << split;

    return s.str();
}

bool DeconvolutionKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::DECONVOLUTION || o.GetType() != KernelType::DECONVOLUTION) {
        return false;
    }

    const deconvolution_params& params = static_cast<const deconvolution_params&>(p);
    const deconvolution_optional_params& optParams = static_cast<const deconvolution_optional_params&>(o);

    bool bSupportedWeightsLayout = params.weights.GetLayout() == GetPreferredWeightsLayout(params);

    const bool bWeightsOK = bSupportedWeightsLayout || optParams.allowStaticInputReordering;

    if (!bWeightsOK) {
        return false;
    }

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

JitConstants DeconvolutionKernelBase::GetJitConstants(const deconvolution_params& dp) const {
    JitConstants jit = WeightBiasKernelBase::GetJitConstants(dp);
    const auto& padding = dp.padding;
    const auto& input = dp.inputs[0];

    int64_t input_offset_with_padding = (int64_t)input.GetFirstElementOffset() -
                                        (dp.filterSize.x - 1 + padding.x) * input.X().pitch -
                                        (dp.filterSize.y - 1 + padding.y) * input.Y().pitch;
    input_offset_with_padding = std::max(input_offset_with_padding, (int64_t)0);

    jit.AddConstants({ MakeJitConstant("STRIDE", dp.stride),
                       MakeJitConstant("PADDING", dp.padding),
                       MakeJitConstant("DILATION", dp.dilation),
                       MakeJitConstant("FILTER_ARRAY_NUM", dp.split),
                       MakeJitConstant("INPUT0_OFFSET_WITH_PADDING", input_offset_with_padding),
                       MakeJitConstant("DEPTHWISE_SEPARABLE_OPT", dp.depthwise_separable_opt),
                       MakeJitConstant("GROUPED", (dp.groups > 1) ? 1 : 0) });
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(dp), "ACCUMULATOR"));
    jit.Merge(MakeTypeJitConstants(GetActivationType(dp), "ACTIVATION"));

    return jit;
}

DeconvolutionKernelBase::DispatchData DeconvolutionKernelBase::SetDefault(const deconvolution_params& params) const {
    auto batch_size = params.output.Batch().v;
    auto output_features = params.output.Feature().v;

    DispatchData kd;

    kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;
    size_t gws0 = output_features * batch_size;
    size_t lws0 = std::min(gws0, static_cast<size_t>(32));
    while (gws0 % lws0) {
        lws0--;
    }
    kd.gws0 = gws0;
    kd.gws1 = params.output.X().v;
    kd.gws2 = params.output.Y().v * params.output.Z().v;
    kd.lws0 = lws0;
    kd.lws1 = 1;
    kd.lws2 = 1;
    kd.efficiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
    return kd;
}

KernelsData DeconvolutionKernelBase::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::DECONVOLUTION);

    if (!Validate(params, options)) {
        return{};
    }

    const deconvolution_params& orgParams = static_cast<const deconvolution_params&>(params);
    DispatchData runInfo = SetDefault(orgParams);
    KernelData kd = KernelData::Default<deconvolution_params>(params);
    deconvolution_params& newParams = *static_cast<deconvolution_params*>(kd.params.get());

    bool succeed = UpdateWeightsParams(newParams,
                                       options,
                                       GetPreferredWeightsLayout(newParams),
                                       kd.weightsReorderParams,
                                       GetSupportedKey(),
                                       newParams.groups);

    if (!succeed) {
        return {};
    }

    auto cldnn_jit = GetJitConstants(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     runInfo,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     DEFAULT,
                     true,
                     !newParams.bias.empty(),
                     1,
                     GetFusedPrimitiveInputsCount(params));
    kernel.arguments.push_back({ArgumentDescriptor::Types::SPLIT, 0});

    kd.estimatedTime = runInfo.efficiency;

    return {kd};
}

Datatype DeconvolutionKernelBase::GetAccumulatorType(const deconvolution_params& params) const {
    if (params.inputs[0].GetDType() == Datatype::INT8 || params.inputs[0].GetDType() == Datatype::UINT8)
        return Datatype::INT32;

    // input is either fp32 or fp16
    // for fp32->fp16 accumulate to fp16, otherwise accumulate to input type
    if (params.output.GetDType() == Datatype::F16)
        return Datatype::F16;

    return params.inputs[0].GetDType();
}

Datatype DeconvolutionKernelBase::GetActivationType(const deconvolution_params& params) const {
    auto accumulator_dt = GetAccumulatorType(params);
    if (accumulator_dt == Datatype::INT32)
        return Datatype::F32;
    return accumulator_dt;
}

}  // namespace kernel_selector
