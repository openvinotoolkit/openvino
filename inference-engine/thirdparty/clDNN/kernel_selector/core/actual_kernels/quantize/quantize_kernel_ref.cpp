// Copyright (c) 2019 Intel Corporation
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


#include <iostream>
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
    CommonDispatchData runInfo;

    auto output = params.output;

    if (output.GetLayout() == DataLayout::b_fs_yx_fsv16 && !params.packed_binary_output) {
        runInfo.gws0 = output.Batch().v;
        runInfo.gws1 = Align(output.Feature().v, sub_group_size);
        runInfo.gws2 = output.Y().v * output.X().v * output.Z().v;

        runInfo.lws0 = 1;
        runInfo.lws1 = sub_group_size;
        runInfo.lws2 = 1;
    } else {
        runInfo.gws0 = output.Batch().v;
        runInfo.gws1 = params.packed_binary_output ? CeilDiv(output.Feature().v, 32) : output.Feature().v;
        runInfo.gws2 = Align(output.X().v * output.Y().v * output.Z().v, 16);

        runInfo.lws0 = 1;
        runInfo.lws1 = 1;
        runInfo.lws2 = 16;
    }

    runInfo.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    return runInfo;
}

JitConstants QuantizeKernelRef::GetJitConstants(const quantize_params& params, const CommonDispatchData& runInfo) const {
    JitConstants jit = Parent::GetJitConstants(params, runInfo);
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

}  // namespace kernel_selector
