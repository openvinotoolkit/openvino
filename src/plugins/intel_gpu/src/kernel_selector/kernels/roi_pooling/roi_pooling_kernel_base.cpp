// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roi_pooling_kernel_base.h"
#include <algorithm>

namespace kernel_selector {

static ROIPoolingKernelBase::DispatchData SetDefault(const roi_pooling_params& params) {
    ROIPoolingKernelBase::DispatchData dispatchData;

    // Determine global work sizes.
    dispatchData.gws[0] = params.outputs[0].LogicalSize();
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;

    // Find largest positive local work size that is divider for global work size.
    dispatchData.lws[0] = std::min(std::max(dispatchData.gws[0], static_cast<size_t>(1)), static_cast<size_t>(32));
    while (dispatchData.gws[0] % dispatchData.lws[0] != 0) {
        --dispatchData.lws[0];
    }
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

JitConstants ROIPoolingKernelBase::GetJitConstants(const roi_pooling_params& rp) const {
    JitConstants jit = MakeBaseParamsJitConstants(rp);

    jit.AddConstants({MakeJitConstant("POOLED_HEIGHT", rp.pooled_height),
                      MakeJitConstant("POOLED_WIDTH", rp.pooled_width),
                      MakeJitConstant("SPATIAL_SCALE", rp.spatial_scale),
                      MakeJitConstant(toString(rp.mode) + "_POOLING", 1)});

    return jit;
}

KernelsData ROIPoolingKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::ROI_POOLING);
    const roi_pooling_params& orgParams = static_cast<const roi_pooling_params&>(params);

    if (!orgParams.activations.empty()) {
        return {};
    }

    DispatchData dispatchData = SetDefault(orgParams);
    KernelData kd = KernelData::Default<roi_pooling_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
    if (orgParams.mode == PoolType::DEFORMABLE_BILINEAR && !orgParams.no_trans)
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});

    return {kd};
}
}  // namespace kernel_selector
