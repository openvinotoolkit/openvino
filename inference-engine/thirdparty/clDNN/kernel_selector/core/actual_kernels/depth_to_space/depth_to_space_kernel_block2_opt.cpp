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

#include "depth_to_space_kernel_block2_opt.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
ParamsKey DepthToSpaceKernelBlock2Opt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    return k;
}

bool DepthToSpaceKernelBlock2Opt::Validate(const Params& p, const optional_params& o) const {
    if (!DepthToSpaceKernelBase::Validate(p, o))
        return false;

    const auto& params = static_cast<const depth_to_space_params&>(p);

    if ((params.block_size != 2) || (params.inputs[0].X().v % 2 != 0))
        return false;

    if (params.mode != DepthToSpaceMode::BLOCKS_FIRST)
        return false;

    return true;
}

CommonDispatchData DepthToSpaceKernelBlock2Opt::SetDefault(const depth_to_space_params& params) const {
    CommonDispatchData runInfo;

    std::vector<size_t> global = { Align(params.inputs[0].X().v / 2, 16),
                                   params.inputs[0].Y().v,
                                   1};

    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    return runInfo;
}

JitConstants DepthToSpaceKernelBlock2Opt::GetJitConstants(const depth_to_space_params& params) const {
    JitConstants jit = Parent::GetJitConstants(params);

    jit.AddConstant(MakeJitConstant("IN_WIDTH", params.inputs[0].X().v / 2));

    return jit;
}

KernelsData DepthToSpaceKernelBlock2Opt::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_5);
}
}  // namespace kernel_selector
