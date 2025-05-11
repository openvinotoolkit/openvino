// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    s << 1;

    return s.str();
}

bool DeconvolutionKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::DECONVOLUTION) {
        return false;
    }

    const deconvolution_params& params = static_cast<const deconvolution_params&>(p);

    bool bSupportedWeightsLayout = params.weights.GetLayout() == GetPreferredWeightsLayout(params);

    const bool bWeightsOK = bSupportedWeightsLayout || params.allowStaticInputReordering;

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
                       MakeJitConstant("FILTER_ARRAY_NUM", 1),
                       MakeJitConstant("INPUT0_OFFSET_WITH_PADDING", input_offset_with_padding),
                       MakeJitConstant("GROUPED", (dp.groups > 1) ? 1 : 0) });
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(dp), "ACCUMULATOR"));
    jit.Merge(MakeTypeJitConstants(GetActivationType(dp), "ACTIVATION"));

    return jit;
}

DeconvolutionKernelBase::DispatchData DeconvolutionKernelBase::SetDefault(const deconvolution_params& params) const {
    auto batch_size = params.outputs[0].Batch().v;
    auto output_features = params.outputs[0].Feature().v;

    DispatchData dispatchData;

    size_t gws0 = output_features * batch_size;
    size_t lws0 = std::min(gws0, static_cast<size_t>(32));
    while (gws0 % lws0) {
        lws0--;
    }

    dispatchData.gws[0] = gws0;
    dispatchData.gws[1] = params.outputs[0].X().v;
    dispatchData.gws[2] = params.outputs[0].Y().v * params.outputs[0].Z().v;

    dispatchData.lws[0] = lws0;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsData DeconvolutionKernelBase::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::DECONVOLUTION);

    if (!Validate(params)) {
        return{};
    }

    const deconvolution_params& orgParams = static_cast<const deconvolution_params&>(params);
    DispatchData dispatchData = SetDefault(orgParams);
    KernelData kd = KernelData::Default<deconvolution_params>(params);
    deconvolution_params& newParams = *static_cast<deconvolution_params*>(kd.params.get());

    bool succeed = UpdateWeightsParams(newParams,
                                       GetPreferredWeightsLayout(newParams),
                                       kd.weightsReorderParams,
                                       GetSupportedKey(),
                                       newParams.groups);

    if (!succeed) {
        return {};
    }

    auto cldnn_jit = GetJitConstants(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     true,
                     !newParams.bias.empty(),
                     1,
                     GetFusedPrimitiveInputsCount(params));

    return {kd};
}

Datatype DeconvolutionKernelBase::GetAccumulatorType(const deconvolution_params& params) const {
    if (params.inputs[0].GetDType() == Datatype::INT8 || params.inputs[0].GetDType() == Datatype::UINT8)
        return Datatype::INT32;

    // input is either fp32 or fp16
    // for fp32->fp16 accumulate to fp16, otherwise accumulate to input type
    if (params.outputs[0].GetDType() == Datatype::F16)
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
