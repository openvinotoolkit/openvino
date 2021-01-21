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

    auto output = params.output;

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
    if (params.output.GetLayout() == DataLayout::b_fs_yx_fsv16 && !params.packed_binary_output) {
        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    }
    return jit;
}

bool QuantizeKernelRef::Validate(const Params& p, const optional_params&) const {
    const quantize_params& params = static_cast<const quantize_params&>(p);
    if (params.inputs.size() != 5)
        return false;

    // Binary packed output is possible only with b_fs_yx_32fp output layout and some input layouts
    if (params.output.GetDType() == Datatype::BINARY &&
        (params.output.GetLayout() != DataLayout::b_fs_yx_32fp ||
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
