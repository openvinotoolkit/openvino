// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "search_sorted_kernel_base.h"

#include <vector>

#include "kernel_selector_utils.h"

namespace kernel_selector {
JitConstants SearchSortedKernelBase::GetJitConstants(const search_sorted_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("RIGHT_MODE", params.right_mode)});

    return jit;
}

void SearchSortedKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const search_sorted_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

SearchSortedKernelBase::DispatchData SearchSortedKernelBase::SetDefault(const search_sorted_params& params) {
    DispatchData dispatchData;
    dispatchData.gws[0] = params.outputs[0].LogicalSize();
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData SearchSortedKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::SEARCH_SORTED);

    const auto& prim_params = static_cast<const search_sorted_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<search_sorted_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(k_data);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     2,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     prim_params.is_shape_agnostic);

    return {k_data};
}
}  // namespace kernel_selector
