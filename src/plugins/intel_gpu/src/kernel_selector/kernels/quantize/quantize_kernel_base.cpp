// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "quantize_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {

bool QuantizeKernelBase::Validate(const Params& p) const {
    const quantize_params& params = static_cast<const quantize_params&>(p);
    if (params.inputs.size() != 5)
        return false;

    return true;
}

JitConstants QuantizeKernelBase::GetJitConstants(const quantize_params& params, const CommonDispatchData& dispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("LEVELS", static_cast<float>(params.levels)));

    jit.AddConstant(MakeJitConstant("LWS_0", dispatchData.lws[0]));
    jit.AddConstant(MakeJitConstant("LWS_1", dispatchData.lws[1]));
    jit.AddConstant(MakeJitConstant("LWS_2", dispatchData.lws[2]));

    return jit;
}

void QuantizeKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const quantize_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData QuantizeKernelBase::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::QUANTIZE);

    KernelData kd = KernelData::Default<quantize_params>(params);
    quantize_params& newParams = *static_cast<quantize_params*>(kd.params.get());

    if (!Validate(params)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams, dispatchData);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];

    kernel.params.workGroups.global = dispatchData.gws;
    kernel.params.workGroups.local = dispatchData.lws;
    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, EXE_MODE_DEFAULT);
    kernel.params.arguments = GetArgsDesc(static_cast<int>(newParams.inputs.size()), false, false, 0, 1, newParams.has_dynamic_tensors());

    return {kd};
}
}  // namespace kernel_selector
