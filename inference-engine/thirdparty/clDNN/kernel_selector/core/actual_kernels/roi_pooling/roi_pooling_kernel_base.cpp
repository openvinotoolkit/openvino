// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "roi_pooling_kernel_base.h"
#include <algorithm>

namespace kernel_selector {

static ROIPoolingKernelBase::DispatchData SetDefault(const roi_pooling_params& params) {
    ROIPoolingKernelBase::DispatchData kd;

    kd.fp16UnitUsed = (params.inputs[0].GetDType() == Datatype::F16);

    // Determine global work sizes.
    kd.gws0 = params.output.LogicalSize();
    kd.gws1 = 1;
    kd.gws2 = 1;

    // Find largest positive local work size that is divider for global work size.
    kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
    while (kd.gws0 % kd.lws0 != 0) {
        --kd.lws0;
    }
    kd.lws1 = 1;
    kd.lws2 = 1;

    return kd;
}

JitConstants ROIPoolingKernelBase::GetJitConstants(const roi_pooling_params& rp) const {
    JitConstants jit = MakeBaseParamsJitConstants(rp);

    jit.AddConstants({MakeJitConstant("POOLED_HEIGHT", rp.pooled_height),
                      MakeJitConstant("POOLED_WIDTH", rp.pooled_width),
                      MakeJitConstant("SPATIAL_SCALE", rp.spatial_scale),
                      MakeJitConstant(toString(rp.mode) + "_POOLING", 1)});

    return jit;
}

KernelsData ROIPoolingKernelBase::GetCommonKernelsData(const Params& params,
                                                       const optional_params& options,
                                                       float estimatedTime) const {
    assert(params.GetType() == KernelType::ROI_POOLING);
    const roi_pooling_params& orgParams = static_cast<const roi_pooling_params&>(params);

    if (!orgParams.activations.empty()) {
        return {};
    }

    DispatchData runInfo = SetDefault(orgParams);
    KernelData kd = KernelData::Default<roi_pooling_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point);
    kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
    if (orgParams.mode == PoolType::DEFORMABLE_BILINEAR && !orgParams.no_trans)
        kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});

    kd.estimatedTime = estimatedTime;

    return {kd};
}
}  // namespace kernel_selector
