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


#include "reorder_kernel_binary.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
ParamsKey ReorderKernelBinary::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::BINARY);
    k.EnableOutputDataType(Datatype::BINARY);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableDifferentTypes();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableOutputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants ReorderKernelBinary::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    KernelData kd = KernelData::Default<reorder_params>(params);
    reorder_params& newParams = *static_cast<reorder_params*>(kd.params.get());

    const auto& input = newParams.inputs[0];
    jit.AddConstant(MakeJitConstant("ELEMENTS_COUNT", input.LogicalSize()));
    jit.AddConstant(MakeJitConstant("IFM_PACK_SIZE", 32));

    if (input.GetDType() == Datatype::BINARY) {
        jit.AddConstant(MakeJitConstant("BINARY_INPUT", 1));
        jit.AddConstant(MakeJitConstant("INPUT_PACKED_FEATURES_NUM", CeilDiv(input.Feature().v, 16)));
    }

    if (params.output.GetDType() == Datatype::BINARY) {
        jit.AddConstant(MakeJitConstant("BINARY_OUTPUT", 1));
        jit.AddConstant(MakeJitConstant("OUTPUT_PACKED_FEATURES_NUM", CeilDiv(params.output.Feature().v, 32)));
    }

    return jit;
}

ReorderKernelBinary::DispatchData ReorderKernelBinary::SetDefault(const reorder_params& params) const {
    DispatchData dispatchData;

    const auto& input = params.inputs[0];

    dispatchData.gws = { input.Batch().v, CeilDiv(input.Feature().v, 32), input.Y().v * input.X().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData ReorderKernelBinary::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::REORDER);

    const reorder_params& orgParams = static_cast<const reorder_params&>(params);

    if (orgParams.inputs[0].GetDType() != Datatype::BINARY &&
        orgParams.output.GetDType() != Datatype::BINARY)
        return {};

    if (orgParams.inputs[0].GetDType() == Datatype::BINARY &&
        orgParams.inputs[0].GetLayout() != DataLayout::b_fs_yx_32fp)
        return {};

    if (orgParams.output.GetDType() == Datatype::BINARY &&
        orgParams.output.GetLayout() != DataLayout::b_fs_yx_32fp)
        return {};

    return GetCommonKernelsData(orgParams, options);
}

KernelsPriority ReorderKernelBinary::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_6;
}
}  // namespace kernel_selector
