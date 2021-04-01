/*
// Copyright (c) 2020 Intel Corporation
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
*/

#include "depth_to_space_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

bool DepthToSpaceKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::DEPTH_TO_SPACE ||
        o.GetType() != KernelType::DEPTH_TO_SPACE) {
        return false;
    }

    const depth_to_space_params& params = static_cast<const depth_to_space_params&>(p);
    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (params.inputs[0].Dimentions() > 5)
        return false;

    return true;
}

CommonDispatchData DepthToSpaceKernelBase::SetDefault(const depth_to_space_params& params) const {
    CommonDispatchData dispatchData;

    dispatchData.gws = { params.output.Batch().v,
                         params.output.Feature().v,
                         params.output.Z().v * params.output.Y().v * params.output.X().v };

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

JitConstants DepthToSpaceKernelBase::GetJitConstants(const depth_to_space_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", params.block_size));
    if (params.mode == DepthToSpaceMode::BLOCKS_FIRST) {
        jit.AddConstant(MakeJitConstant("BLOCKS_FIRST", 1));
    } else {
        jit.AddConstant(MakeJitConstant("DEPTH_FIRST", 1));
    }

    return jit;
}

KernelsData DepthToSpaceKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    KernelData kd = KernelData::Default<depth_to_space_params>(params);
    depth_to_space_params& newParams = *static_cast<depth_to_space_params*>(kd.params.get());

    if (!Validate(params, options)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     DEFAULT, false, false, 1, GetFusedPrimitiveInputsCount(params));

    return { kd };
}
}  // namespace kernel_selector
