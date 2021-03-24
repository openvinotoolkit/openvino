// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "border_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {
JitConstants BorderKernelBase::GetJitConstants(const border_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("LT_SIZES", params.lt_sizes),
                      MakeJitConstant("RB_SIZES", params.rb_sizes),
                      MakeJitConstant("BORDER_VALUE", params.border_value),
                      MakeJitConstant(toString(params.b_type), "")});

    return jit;
}

BorderKernelBase::DispatchData BorderKernelBase::SetDefault(const border_params& params) const {
    const auto& output = params.output;

    DispatchData dispatchData;

    dispatchData.gws = { output.X().v * output.Z().v, output.Y().v * output.W().v, output.Batch().v * output.Feature().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData BorderKernelBase::GetCommonKernelsData(const Params& params,
                                                   const optional_params& options) const {
    assert(params.GetType() == KernelType::BORDER);

    const auto& prim_params =
        static_cast<const border_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<border_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    return {k_data};
}
}  // namespace kernel_selector
