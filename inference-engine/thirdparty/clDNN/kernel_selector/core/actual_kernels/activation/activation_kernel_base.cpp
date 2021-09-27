// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ActivationKernelBase::DispatchData ActivationKernelBase::SetDefault(const activation_params& arg) const {
    const auto& out = arg.output;

    DispatchData dispatchData;
    if (out.GetLayout() == DataLayout::yxfb) {
        dispatchData.gws = {out.Feature().v * out.Batch().v, out.X().v, out.Y().v};
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo);
    } else if (out.GetLayout() == DataLayout::b_fs_yx_fsv16 || out.GetLayout() == DataLayout::b_fs_yx_fsv32) {
        dispatchData.gws = {Align(out.Feature().v, 16) * out.Batch().v, out.X().v, out.Y().v};
        dispatchData.lws = {16, 1, 1};
    } else if (out.GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16 || out.GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv32) {
        dispatchData.gws = {out.X().v * out.Y().v, Align(out.Feature().v, 16), Align(out.Batch().v, 16)};
        dispatchData.lws = {1, 16, 16};
    } else {
        dispatchData.gws = {out.X().v, out.Y().v * out.Z().v, out.Feature().v * out.Batch().v};
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo);
    }

    return dispatchData;
}

JitConstants ActivationKernelBase::GetJitConstants(const activation_params& params, DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& inputNlParams = params.inputActivationParams;

    jit.AddConstants({
        MakeJitConstant("PARAMS_NUM", GetActivationAdditionalParamsNumber(params.activations[0].function)),
    });

    if (!inputNlParams.empty()) {
        jit.AddConstants({
            MakeJitConstant("ADDITIONAL_PARAMS", inputNlParams[0]),
            MakeJitConstant("PARAMETERIZED", ""),
        });
    }

    return jit;
}

bool ActivationKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::ACTIVATION ||
        o.GetType() != KernelType::ACTIVATION) {
        return false;
    }
    const activation_params& orgParams = static_cast<const activation_params&>(p);

    for (auto& fused_op : orgParams.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

KernelsData ActivationKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<activation_params>(params);

    activation_params& newParams = *static_cast<activation_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams);
    auto cldnn_jit = GetJitConstants(newParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     DEFAULT, false, false, 1, GetFusedPrimitiveInputsCount(params));

    if (!newParams.inputActivationParams.empty()) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SLOPE, 0});
    }

    return {kd};
}
}  // namespace kernel_selector
