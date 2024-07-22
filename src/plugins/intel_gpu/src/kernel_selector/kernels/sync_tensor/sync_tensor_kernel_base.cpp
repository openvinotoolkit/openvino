// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sync_tensor_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
JitConstants SyncTensorKernelBase::GetJitConstants(const sync_tensor_params& params,
                                                SyncTensorKernelBase::DispatchData dispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("ALONG_" + toString(params.dim), "1")});

    return jit;
}

SyncTensorKernelBase::DispatchData SyncTensorKernelBase::SetDefault(const sync_tensor_params&) const {
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
    dispatchData.maxSlmSize = 0;

    return dispatchData;
}

bool SyncTensorKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SYNC_TENSOR) {
        return false;
    }

    return true;
}

KernelsData SyncTensorKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const sync_tensor_params& orgParams = static_cast<const sync_tensor_params&>(params);
    KernelData kd = KernelData::Default<sync_tensor_params>(params);

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
}  // namespace kernel_selector
