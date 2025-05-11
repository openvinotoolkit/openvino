// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize_kernel_base.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
JitConstants NormalizeKernelBase::GetJitConstants(const normalize_params& np) const {
    JitConstants jit = MakeBaseParamsJitConstants(np);

    jit.AddConstants({
        MakeJitConstant("SCALE_TABLE", np.scaleTable),
        MakeJitConstant("EPSILON", np.epsilon),
        MakeJitConstant(toString(np.normMode), ""),
        MakeJitConstant("THRESHOLD", 0.0001f),
    });

    auto activation_dt = GetActivationType(np);
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    if (!np.fused_ops.empty()) {
        std::vector<std::string> idx_order = { "b", "f", "y", "x" };
        auto conf = FusedOpsConfiguration("", idx_order, "result", activation_dt);
        jit.Merge(MakeFusedOpsJitConstants(np, { conf }));
    }

    return jit;
}

NormalizeKernelBase::DispatchData NormalizeKernelBase::SetDefault(const normalize_params& params) const {
    const auto& output = params.outputs[0];

    DispatchData dispatchData;
    if (params.normMode == NormalizeMode::WITHIN_SPATIAL) {
        dispatchData.gws = {output.X().v, output.Y().v, output.Batch().v};
    } else {
        dispatchData.gws = {output.Batch().v, 1, 1};
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData NormalizeKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::NORMALIZE);
    if (!Validate(params))
        return {};

    const normalize_params& orgParams = static_cast<const normalize_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<normalize_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params));

    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALE_TABLE, 0});

    return {kd};
}

bool NormalizeKernelBase::Validate(const Params& params) const {
    const normalize_params& orgParams = static_cast<const normalize_params&>(params);

    for (auto& fused_op : orgParams.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

Datatype NormalizeKernelBase::GetActivationType(const normalize_params& params) const {
    if (params.outputs[0].GetDType() == Datatype::F16)
        return Datatype::F16;
    return Datatype::F32;
}
}  // namespace kernel_selector
