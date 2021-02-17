// Copyright (c) 2019-2020 Intel Corporation
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
    dispatchData.gws = { params.output.Y().v,        // beam
                         params.output.Feature().v,  // batch
                         1 };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}

KernelsData GatherTreeKernelBase::GetCommonKernelsData(const Params& params,
                                                       const optional_params& options) const {
    assert(params.GetType() == KernelType::GATHER_TREE);
    const auto& gt_params = static_cast<const gather_tree_params&>(params);

    auto dispatchData = SetDefault(gt_params);
    auto kernel_data = KernelData::Default<gather_tree_params>(params);
    auto cldnn_jit = GetJitConstants(gt_params);
    auto entry_point = GetEntryPoint(kernelName, gt_params.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);
    FillCLKernelData(kernel_data.kernels[0],
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     DEFAULT,
                     false,
                     false,
                     static_cast<int>(gt_params.inputs.size()));
    return { kernel_data };
}
}  // namespace kernel_selector
