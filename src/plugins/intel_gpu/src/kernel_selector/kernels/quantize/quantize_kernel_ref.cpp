// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quantize_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>

static const size_t sub_group_size = 32;

namespace kernel_selector {
ParamsKey QuantizeKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

CommonDispatchData QuantizeKernelRef::SetDefault(const quantize_params& params) const {
    CommonDispatchData dispatchData;

    auto output = params.outputs[0];

    if (output.GetLayout() == DataLayout::b_fs_yx_fsv16) {
        dispatchData.gws[0] = output.Batch().v;
        dispatchData.gws[1] = Align(output.Feature().v, sub_group_size);
        dispatchData.gws[2] = output.Y().v * output.X().v * output.Z().v;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = sub_group_size;
        dispatchData.lws[2] = 1;
    } else {
        dispatchData.gws[0] = output.Batch().v;
        dispatchData.gws[1] = output.Feature().v;
        dispatchData.gws[2] = Align(output.X().v * output.Y().v * output.Z().v * output.W().v * output.U().v * output.V().v, 16);

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 16;
    }

    return dispatchData;
}

JitConstants QuantizeKernelRef::GetJitConstants(const quantize_params& params, const CommonDispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);
    if (params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16) {
        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    }
    return jit;
}

bool QuantizeKernelRef::Validate(const Params& p) const {
    const quantize_params& params = static_cast<const quantize_params&>(p);
    if (params.inputs.size() != 5)
        return false;

    return true;
}

KernelsPriority QuantizeKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
