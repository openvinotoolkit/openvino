// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel_fs_b_yx_fsv32_to_bfyx.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {

static const size_t fsv = 32;
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

    size_t optimal_size = GetOptimalSize(Align(params.outputs[0].X().v, x_block_align), optimal_x_sizes);
    jit.AddConstant(MakeJitConstant("FSV", fsv));
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", optimal_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKED_SIZE", Align(params.outputs[0].X().v, x_block_align) / optimal_size));

    size_t lws1 = GetOptimalSize(Align(params.outputs[0].Feature().v, fsv), optimal_feature_sizes);

    jit.AddConstant(MakeJitConstant("LWS1", lws1));

    if (params.outputs[0].Feature().v % fsv != 0) {
        jit.AddConstant(MakeJitConstant("LEFTOVERS_OC", params.outputs[0].Feature().v % fsv));
    }

    if (params.outputs[0].X().v % x_block_align != 0) {
        jit.AddConstant(MakeJitConstant("LEFTOVERS_OX", params.outputs[0].X().v % x_block_align));
    }

    return jit;
}

ReorderKernelBase::DispatchData ReorderKernel_fs_b_yx_fsv32_to_bfyx::SetDefault(const reorder_params& params) const {
    DispatchData dispatchData;

    auto x_aligned = Align(params.outputs[0].X().v, x_block_align);

    dispatchData.gws[0] = params.outputs[0].Batch().v;
    dispatchData.gws[1] = Align(params.outputs[0].Feature().v, fsv);
    dispatchData.gws[2] = params.outputs[0].Y().v * x_aligned / GetOptimalSize(x_aligned, optimal_x_sizes);

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = GetOptimalSize(dispatchData.gws[1], optimal_feature_sizes);
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsData ReorderKernel_fs_b_yx_fsv32_to_bfyx::GetKernelsData(const Params& params) const {
    const auto& orgParams = static_cast<const reorder_params&>(params);
    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderKernel_fs_b_yx_fsv32_to_bfyx::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}
}  // namespace kernel_selector
