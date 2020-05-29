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
#include "quantize_kernel_scale_shift_opt.h"
#include "kernel_selector_utils.h"
#include <string>

static const size_t sub_group_size = 32;

namespace kernel_selector {
ParamsKey QuantizeKernelScaleShift::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableInputLayout(DataLayout::byxf_af32);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::byxf_af32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableQuantizeScaleShiftOpt();
    return k;
}

CommonDispatchData QuantizeKernelScaleShift::SetDefault(const quantize_params& params, const optional_params&) const {
    CommonDispatchData runInfo;

    auto output = params.output;

    if (output.GetLayout() == DataLayout::b_fs_yx_fsv16) {
        runInfo.gws0 = output.Y().v * output.X().v;
        runInfo.gws1 = Align(output.Feature().v, sub_group_size);
        runInfo.gws2 = output.Batch().v;

        runInfo.lws0 = 1;
        runInfo.lws1 = sub_group_size;
        runInfo.lws2 = 1;
    } else {
        auto global = GetTensorFriendlyWorkGroups(output);
        auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

        runInfo.gws0 = global[0];
        runInfo.gws1 = global[1];
        runInfo.gws2 = global[2];

        runInfo.lws0 = local[0];
        runInfo.lws1 = local[1];
        runInfo.lws2 = local[2];
    }

    runInfo.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    return runInfo;
}

JitConstants QuantizeKernelScaleShift::GetJitConstants(const quantize_params& params, const CommonDispatchData& runInfo) const {
    JitConstants jit = Parent::GetJitConstants(params, runInfo);

    if (params.output.GetLayout() == DataLayout::b_fs_yx_fsv16) {
        jit.AddConstant(MakeJitConstant("GWS_BATCH", 2));
        jit.AddConstant(MakeJitConstant("GWS_FEATURE", 1));
        jit.AddConstant(MakeJitConstant("GWS_YX", 0));
        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    } else {
        auto tensor_jits = GetTensorFriendlyWorkGroupsJit(params.output);
        jit.Merge(tensor_jits);
    }

    jit.AddConstant(MakeJitConstant("HAS_POST_SCALE", params.has_post_scale));
    jit.AddConstant(MakeJitConstant("HAS_POST_SHIFT", params.has_post_shift));
    jit.AddConstant(MakeJitConstant("HAS_PRE_SHIFT", params.has_pre_shift));
    jit.AddConstant(MakeJitConstant("HAS_CLAMP", params.has_clamp));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_INPUT_RANGE", params.per_tensor_input_range));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_INPUT_SCALE", params.per_tensor_input_scale));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_INPUT_SHIFT", params.per_tensor_input_shift));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_OUTPUT_SCALE", params.per_tensor_output_scale));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_OUTPUT_SHIFT", params.per_tensor_output_shift));
    jit.AddConstant(MakeJitConstant("IN_LO_VAL", params.in_lo));
    jit.AddConstant(MakeJitConstant("IN_HI_VAL", params.in_hi));
    jit.AddConstant(MakeJitConstant("IN_SCALE_VAL", params.in_scale));
    jit.AddConstant(MakeJitConstant("IN_SHIFT_VAL", params.in_shift));
    jit.AddConstant(MakeJitConstant("OUT_SCALE_VAL", params.out_scale));
    jit.AddConstant(MakeJitConstant("OUT_SHIFT_VAL", params.out_shift));

    return jit;
}

bool QuantizeKernelScaleShift::Validate(const Params& p, const optional_params&) const {
    const quantize_params& params = static_cast<const quantize_params&>(p);
    if (params.inputs.size() != 9)
        return false;

    return true;
}

}  // namespace kernel_selector
