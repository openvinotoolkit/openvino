// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_kernel_base.h"

namespace kernel_selector {
JitConstants SoftmaxKernelBase::GetJitConstants(const softmax_params& params,
                                                SoftmaxKernelBase::DispatchData dispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("ALONG_" + toString(params.dim), "1")});

    return jit;
}

SoftmaxKernelBase::DispatchData SoftmaxKernelBase::SetDefault(const softmax_params&) const {
    DispatchData dispatchData;

    dispatchData.gws[0] = 1;
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    dispatchData.leftovers = 0;
    dispatchData.itemsNum = 0;
    dispatchData.normIndex = 0;
    dispatchData.dataSetsCount = 0;
    dispatchData.dataSetSize = 0;

    return dispatchData;
}

bool SoftmaxKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SOFT_MAX) {
        return false;
    }

    return true;
}

KernelsData SoftmaxKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const softmax_params& orgParams = static_cast<const softmax_params&>(params);
    KernelData kd = KernelData::Default<softmax_params>(params);

    auto dispatchData = SetDefault(orgParams);
    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     orgParams.is_shape_agnostic);

    return {kd};
}

bool SoftmaxKernelBaseBF::Validate(const Params& p) const {
    if (!Parent::Validate(p)) {
        return false;
    }

    const softmax_params& params = static_cast<const softmax_params&>(p);
    const auto& input = params.inputs[0];

    if (input.GetLayout() == DataLayout::bf || input.GetLayout() == DataLayout::fb) {
        return true;
    }

    switch (params.dim) {
        case SoftmaxDim::X:
            return ((!input.Y().is_dynamic && input.Y().v == 1) || input.GetLayout() == DataLayout::bfyx) &&
                   !input.Z().is_dynamic && input.Z().v == 1 &&
                   ((!input.Feature().is_dynamic && input.Feature().v == 1) || input.GetLayout() == DataLayout::bfyx);
        case SoftmaxDim::Y:
            return !input.X().is_dynamic && input.X().v == 1 &&
                   !input.Z().is_dynamic && input.Z().v == 1 &&
                   ((!input.Feature().is_dynamic && input.Feature().v == 1) || input.GetLayout() == DataLayout::bfyx);
        case SoftmaxDim::Z:
            return !input.X().is_dynamic && input.X().v == 1 &&
                   !input.Y().is_dynamic && input.Y().v == 1 &&
                   !input.Feature().is_dynamic && input.Feature().v == 1;
        case SoftmaxDim::FEATURE:
            return !input.X().is_dynamic && input.X().v == 1 &&
                   !input.Y().is_dynamic && input.Y().v == 1 &&
                   !input.Z().is_dynamic && input.Z().v == 1;
        default:
            return false;
    }
}

SoftmaxKernelBase::DispatchData SoftmaxKernelBaseBF::SetDefault(const softmax_params& params) const {
    const auto& input = params.inputs[0];

    DispatchData dispatchData = Parent::SetDefault(params);

    if (params.dim == SoftmaxDim::Y && input.Feature().v > 1 && input.GetLayout() == DataLayout::bfyx) {
        // Flatten BF for such case, X is expected to be 1
        OPENVINO_ASSERT(input.X().v == 1, "[GPU] SoftmaxKernelBaseBF: input.X() is expected to be 1 while actual value is ", input.X().v);
        dispatchData.dataSetSize = input.Y().v;
        dispatchData.dataSetsCount = input.Batch().v * input.Feature().v;
    } else if (params.dim == SoftmaxDim::X && (input.Feature().v > 1 || input.Y().v > 1) && input.GetLayout() == DataLayout::bfyx) {
        // Flatten BFY for such case
        dispatchData.dataSetSize = input.X().v;
        dispatchData.dataSetsCount = input.Batch().v * input.Feature().v * input.Y().v;
    } else {
        auto flatten_input = input.FlattenFeatureAndSpatials();
        dispatchData.dataSetSize = flatten_input.Feature().v;
        dispatchData.dataSetsCount = input.Batch().v;
    }

    return dispatchData;
}
}  // namespace kernel_selector
