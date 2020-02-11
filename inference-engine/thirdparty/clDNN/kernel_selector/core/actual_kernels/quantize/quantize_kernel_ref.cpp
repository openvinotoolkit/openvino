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
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfyx_f16);
    k.EnableInputLayout(DataLayout::bfzyx_f16);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfyx_f16);
    k.EnableOutputLayout(DataLayout::bfzyx_f16);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::byxf_af32);
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

    runInfo.gws0 = output.Batch().v;
    runInfo.gws1 = params.packed_binary_output ? CeilDiv(output.Feature().v, 32) : output.Feature().v;
    runInfo.gws2 = Align(output.X().v * output.Y().v * output.Z().v, 16);

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = 16;

    runInfo.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    return runInfo;
}

JitConstants QuantizeKernelRef::GetJitConstants(const quantize_params& params) const {
    JitConstants jit = Parent::GetJitConstants(params);
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
         params.inputs[0].GetLayout() != DataLayout::bfzyx_f16 &&
         params.inputs[0].GetLayout() != DataLayout::bfyx_f16 &&
         params.inputs[0].GetLayout() != DataLayout::fs_b_yx_fsv32)))
        return false;
    return true;
}

}  // namespace kernel_selector
