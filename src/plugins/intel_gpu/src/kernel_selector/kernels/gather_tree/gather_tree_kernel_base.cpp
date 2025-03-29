// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_tree_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {
JitConstants GatherTreeKernelBase::GetJitConstants(const gather_tree_params & params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    return jit;
}

GatherTreeKernelBase::DispatchData GatherTreeKernelBase::SetDefault(const gather_tree_params & params) const {
    DispatchData dispatchData;
    /*
        b -> time
        f -> batch
        y -> beam
    */
    dispatchData.gws = { params.outputs[0].Y().v,        // beam
                         params.outputs[0].Feature().v,  // batch
                         1 };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}

KernelsData GatherTreeKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::GATHER_TREE);
    const auto& gt_params = static_cast<const gather_tree_params&>(params);

    auto dispatchData = SetDefault(gt_params);
    auto kernel_data = KernelData::Default<gather_tree_params>(params);
    auto cldnn_jit = GetJitConstants(gt_params);
    auto entry_point = GetEntryPoint(kernelName, gt_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);
    FillCLKernelData(kernel_data.kernels[0],
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     static_cast<int>(gt_params.inputs.size()));
    return { kernel_data };
}
}  // namespace kernel_selector
