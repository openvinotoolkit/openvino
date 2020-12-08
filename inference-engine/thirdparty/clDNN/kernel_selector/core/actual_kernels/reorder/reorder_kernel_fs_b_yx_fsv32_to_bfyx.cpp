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

#include "reorder_kernel_fs_b_yx_fsv32_to_bfyx.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {

static const size_t fsv = 32;
static const size_t sub_group_size = 16;
static const size_t x_block_align = 8;
static const std::vector<size_t> optimal_x_sizes = { 16, 8, 4, 2, 1 };
static const std::vector<size_t> optimal_feature_sizes = { 16, 8, 1 };

static size_t GetOptimalSize(size_t val, std::vector<size_t> optimal_sizes) {
    for (auto& s : optimal_sizes)
        if (val % s == 0)
            return s;
    return 1;
}

ParamsKey ReorderKernel_fs_b_yx_fsv32_to_bfyx::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants ReorderKernel_fs_b_yx_fsv32_to_bfyx::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));

    size_t optimal_size = GetOptimalSize(Align(params.output.X().v, x_block_align), optimal_x_sizes);
    jit.AddConstant(MakeJitConstant("FSV", fsv));
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", optimal_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKED_SIZE", Align(params.output.X().v, x_block_align) / optimal_size));

    size_t lws1 = GetOptimalSize(Align(params.output.Feature().v, fsv), optimal_feature_sizes);

    jit.AddConstant(MakeJitConstant("LWS1", lws1));

    if (params.output.Feature().v % fsv != 0) {
        jit.AddConstant(MakeJitConstant("LEFTOVERS_OC", params.output.Feature().v % fsv));
    }

    if (params.output.X().v % x_block_align != 0) {
        jit.AddConstant(MakeJitConstant("LEFTOVERS_OX", params.output.X().v % x_block_align));
    }

    return jit;
}

ReorderKernelBase::DispatchData ReorderKernel_fs_b_yx_fsv32_to_bfyx::SetDefault(const reorder_params& params) const {
    DispatchData dispatchData;

    auto x_aligned = Align(params.output.X().v, x_block_align);

    dispatchData.gws[0] = params.output.Batch().v;
    dispatchData.gws[1] = Align(params.output.Feature().v, fsv);
    dispatchData.gws[2] = params.output.Y().v * x_aligned / GetOptimalSize(x_aligned, optimal_x_sizes);

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = GetOptimalSize(dispatchData.gws[1], optimal_feature_sizes);
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsData ReorderKernel_fs_b_yx_fsv32_to_bfyx::GetKernelsData(const Params& params, const optional_params& options) const {
    const auto& orgParams = static_cast<const reorder_params&>(params);
    return GetCommonKernelsData(orgParams, options, FORCE_PRIORITY_2);
}
}  // namespace kernel_selector
