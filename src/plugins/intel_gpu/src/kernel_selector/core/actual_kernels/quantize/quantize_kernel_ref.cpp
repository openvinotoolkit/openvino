// Copyright (C) 2018-2022 Intel Corporation
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
    k.EnableOutputDataType(Datatype::BINARY);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableQuantizePackedBinaryOutput();
    return k;
}

CommonDispatchData QuantizeKernelRef::SetDefault(const quantize_params& params, const optional_params&) const {
    CommonDispatchData dispatchData;

    auto output = params.outputs[0];

    if (output.GetLayout() == DataLayout::b_fs_yx_fsv16 && !params.packed_binary_output) {
        dispatchData.gws[0] = output.Batch().v;
        dispatchData.gws[1] = Align(output.Feature().v, sub_group_size);
        dispatchData.gws[2] = output.Y().v * output.X().v * output.Z().v;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = sub_group_size;
        dispatchData.lws[2] = 1;
    } else {
        dispatchData.gws[0] = output.Batch().v;
        dispatchData.gws[1] = params.packed_binary_output ? CeilDiv(output.Feature().v, 32) : output.Feature().v;
        dispatchData.gws[2] = Align(output.X().v * output.Y().v * output.Z().v * output.W().v, 16);

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 16;
    }

    return dispatchData;
}

JitConstants QuantizeKernelRef::GetJitConstants(const quantize_params& params, const CommonDispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);
    if (params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16 && !params.packed_binary_output) {
        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    }
    return jit;
}

bool QuantizeKernelRef::Validate(const Params& p, const optional_params&) const {
    const quantize_params& params = static_cast<const quantize_params&>(p);
    if (params.inputs.size() != 5)
        return false;

    // Binary packed output is possible only with b_fs_yx_32fp output layout and some input layouts
    if (params.outputs[0].GetDType() == Datatype::BINARY &&
        (params.outputs[0].GetLayout() != DataLayout::b_fs_yx_32fp ||
        (params.inputs[0].GetLayout() != DataLayout::bfyx &&
         params.inputs[0].GetLayout() != DataLayout::bfzyx &&
         params.inputs[0].GetLayout() != DataLayout::b_fs_zyx_fsv16 &&
         params.inputs[0].GetLayout() != DataLayout::b_fs_yx_fsv16 &&
         params.inputs[0].GetLayout() != DataLayout::fs_b_yx_fsv32)))
        return false;
    return true;
}

KernelsPriority QuantizeKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
