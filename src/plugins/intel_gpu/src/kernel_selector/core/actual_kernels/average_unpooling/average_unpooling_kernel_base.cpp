// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "average_unpooling_kernel_base.h"
#include <algorithm>

namespace kernel_selector {
bool AverageUnpoolingKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::AVERAGE_UNPOOLING || o.GetType() != KernelType::AVERAGE_UNPOOLING) {
        return false;
    }

    return true;
}

JitConstants AverageUnpoolingKernelBase::GetJitConstants(const average_unpooling_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("UNPOOL", params.unpoolSize), MakeJitConstant("STRIDE", params.unpoolStride)});

    return jit;
}

AverageUnpoolingKernelBase::DispatchData AverageUnpoolingKernelBase::SetDefault(
    const average_unpooling_params& params) const {
    const auto& input = params.inputs[0];

    DispatchData dispatchData;

    if (input.GetLayout() == DataLayout::bfyx || input.GetLayout() == DataLayout::byxf) {
        // Determine global work sizes.
        dispatchData.gws =  { Align(input.X().v, 32),               // X
                              input.Y().v,                          // Y
                              input.Batch().v * input.Feature().v,  // B, F
                            };

        dispatchData.lws = { 32, 1, 1 };
    } else {
        // Determine global work sizes.
        dispatchData.gws = { input.Batch().v * input.Feature().v,  // B, F
                             input.X().v,                          // X
                             input.Y().v };                        // Y

        dispatchData.lws = {1, 1, 1};
        dispatchData.lws[0] = std::min(std::max(dispatchData.gws[0], static_cast<size_t>(1)), static_cast<size_t>(32));
        while (dispatchData.gws[0] % dispatchData.lws[0] != 0) {
            --dispatchData.lws[0];
        }
    }

    return dispatchData;
}

KernelsData AverageUnpoolingKernelBase::GetCommonKernelsData(const Params& params,
                                                             const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const average_unpooling_params& orgParams = static_cast<const average_unpooling_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<average_unpooling_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    return {kd};
}
}  // namespace kernel_selector
