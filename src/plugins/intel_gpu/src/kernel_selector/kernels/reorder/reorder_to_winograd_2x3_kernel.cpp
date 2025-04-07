// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_to_winograd_2x3_kernel.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderToWinograd2x3Kernel::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::winograd_2x3_s1_data);
    k.EnableWinogradReorder();
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants ReorderToWinograd2x3Kernel::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);

    jit.AddConstant(MakeJitConstant("INPUT0_OFFSET_SIZE_X", params.winograd_input_offset_x));
    jit.AddConstant(MakeJitConstant("INPUT0_OFFSET_SIZE_Y", params.winograd_input_offset_y));

    return jit;
}

ReorderToWinograd2x3Kernel::DispatchData ReorderToWinograd2x3Kernel::SetDefault(const reorder_params& params) const {
    DispatchData dispatchData;

    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    dispatchData.gws[0] = static_cast<size_t>(input.Feature().v * input.Batch().v);
    dispatchData.gws[1] = static_cast<size_t>(params.winograd_nr_tiles_x);
    dispatchData.gws[2] = static_cast<size_t>(output.Y().v);

    dispatchData.lws[0] = input.Feature().v > 32 ? 32 : static_cast<size_t>(input.Feature().v);
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsData ReorderToWinograd2x3Kernel::GetKernelsData(const Params& params) const {
    const reorder_params& orgParams = static_cast<const reorder_params&>(params);
    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderToWinograd2x3Kernel::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_6;
}
}  // namespace kernel_selector
