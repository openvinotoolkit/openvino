// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_from_winograd_2x3_kernel.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderFromWinograd2x3Kernel::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::winograd_2x3_s1_data);
    k.EnableWinogradReorder();
    k.EnableDifferentTypes();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants ReorderFromWinograd2x3Kernel::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);

    constexpr auto output_tile_width = 2;  // by definition of F(2,3)

    if (params.outputs[0].X().v % output_tile_width != 0)
        jit.AddConstant(MakeJitConstant("LEFTOVERS", 1));

    return jit;
}

ReorderFromWinograd2x3Kernel::DispatchData ReorderFromWinograd2x3Kernel::SetDefault(
    const reorder_params& params) const {
    DispatchData dispatchData;

    constexpr auto output_tile_width = 2;  // by definition of F(2,3)
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    dispatchData.gws[0] = static_cast<size_t>(output.Feature().v * output.Batch().v);
    dispatchData.gws[1] = static_cast<size_t>(output.X().v / output_tile_width);
    dispatchData.gws[2] = static_cast<size_t>(output.Y().v);

    dispatchData.lws[0] = input.Feature().v > 32 ? 32 : static_cast<size_t>(input.Feature().v);
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsData ReorderFromWinograd2x3Kernel::GetKernelsData(const Params& params) const {
    const reorder_params& orgParams = static_cast<const reorder_params&>(params);
    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderFromWinograd2x3Kernel::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_6;
}
}  // namespace kernel_selector
