// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eltwise_kernel_portable.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "kernel_selector_utils.h"

namespace kernel_selector {

bool EltwiseKernelPortable::IsDense(const eltwise_params& params) const {
    return !params.is_shape_agnostic && !params.has_dynamic_tensors() && !params.broadcast && !params.layoutBased && params.stride.empty() &&
           params.fused_ops.empty() && params.updateInputIds.empty() && params.engineInfo.maxWorkGroupSize != 0 && params.outputs[0].LogicalSize() != 0 &&
           params.outputs[0].LogicalSize() <= std::numeric_limits<uint32_t>::max() && params.outputs[0].PhysicalSize() == params.outputs[0].LogicalSize() &&
           CheckInputsOutputNoPitchSameDims(params);
}

EltwiseKernelBase::DispatchData EltwiseKernelPortable::SetDefault(const eltwise_params& params) const {
    if (!IsDense(params)) {
        return EltwiseKernelRef::SetDefault(params);
    }
    const auto& info = params.engineInfo;
    const auto count = params.outputs[0].LogicalSize();
    const auto local = std::max<size_t>(1, std::min(count, static_cast<size_t>(info.maxWorkGroupSize)));
    const auto groups = CeilDiv(count, local);
    // A balanced grid avoids concentrating all workgroups on one dispatch axis.
    const auto columns = static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(groups))));
    DispatchData dispatch;
    dispatch.gws = {columns * local, CeilDiv(groups, columns), 1};
    dispatch.lws = {local, 1, 1};
    return dispatch;
}

JitConstants EltwiseKernelPortable::GetJitConstants(const eltwise_params& params) const {
    auto jit = EltwiseKernelRef::GetJitConstants(params);
    const auto dense = IsDense(params);
    jit.AddConstant(MakeJitConstant("ELTWISE_PORTABLE_DENSE", dense));
    if (dense) {
        const auto dispatch = SetDefault(params);
        const auto count = params.outputs[0].LogicalSize();
        jit.AddConstant(MakeJitConstant("ELTWISE_ROW_SIZE", dispatch.gws[0]));
        jit.AddConstant(MakeJitConstant("ELTWISE_HAS_TAIL", count != dispatch.gws[0] * dispatch.gws[1]));
        jit.AddConstant(MakeJitConstant("ELTWISE_ELEMENTS_COUNT", count));
    }
    return jit;
}

}  // namespace kernel_selector
