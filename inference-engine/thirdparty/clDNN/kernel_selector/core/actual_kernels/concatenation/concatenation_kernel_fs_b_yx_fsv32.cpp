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
#include <algorithm>
#include "concatenation_kernel_fs_b_yx_fsv32.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

static constexpr size_t subGroupSize = 16;
static constexpr size_t fsv = 32;
static constexpr size_t fsvPerThread = fsv / subGroupSize;

ParamsKey ConcatenationKernel_fs_b_yx_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableConcatAxis(ConcatAxis::FEATURE);
    k.EnableConcatKernelPerInput();
    return k;
}

bool ConcatenationKernel_fs_b_yx_fsv32::Validate(const Params& p, const optional_params& o) const {
    if (!ConcatenationKernelBase::Validate(p, o)) {
        return false;
    }

    const concatenation_params& params = static_cast<const concatenation_params&>(p);

    if (params.axis != ConcatAxis::FEATURE)
        return false;

    // all inputs have to have same layout
    auto same_layout = params.inputs[0].GetLayout();
    for (const auto& lt : params.inputs) {
        if (lt.GetLayout() != same_layout) {
            return false;
        }
    }

    return true;
}

ConcatenationKernelBase::DispatchData ConcatenationKernel_fs_b_yx_fsv32::SetDefault(const concatenation_params& params) const {
    DispatchData runInfo = ConcatenationKernelBase::SetDefault(params);
    const auto& input = params.inputs[0];

    runInfo.gws0 = input.X().v;
    runInfo.gws1 = input.Y().v;
    runInfo.gws2 = CeilDiv(input.Feature().v, fsv) * subGroupSize * input.Batch().v;

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = subGroupSize;

    runInfo.effiency = FORCE_PRIORITY_1;

    return runInfo;
}

JitConstants ConcatenationKernel_fs_b_yx_fsv32::GetJitConstants(const concatenation_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("ALIGNED", params.isAligned));
    jit.AddConstant(MakeJitConstant("FSV", fsv));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subGroupSize));
    jit.AddConstant(MakeJitConstant("FSV_PER_THREAD", fsvPerThread));

    return jit;
}

KernelsData ConcatenationKernel_fs_b_yx_fsv32::GetKernelsData(const Params& params, const optional_params& optParams) const {
    if (!Validate(params, optParams)) {
        return {};
    }

    const concatenation_params& orgParams = static_cast<const concatenation_params&>(params);

    KernelData kd = KernelData::Default<concatenation_params>(params, orgParams.inputs.size());

    uint32_t lastOffset = 0;
    const auto concatChannelIndex = GetConcatChannelIndex(orgParams);
    float effiency = FORCE_PRIORITY_1;
    size_t ifm_offset = 0;
    for (size_t i = 0; i < orgParams.inputs.size(); i++) {
        const auto& input = orgParams.inputs[i];

        auto newParams = orgParams;
        newParams.inputs.resize(1);
        newParams.inputs[0] = input;
        size_t ifm = input.Feature().v;
        newParams.isAligned = ifm_offset % fsv == 0;
        ifm_offset += ifm;

        auto& kernel = kd.kernels[i];
        DispatchData runInfo = SetDefault(newParams);
        auto cldnnJit = GetJitConstants(newParams);
        auto entryPoint = GetEntryPoint(kernelName, newParams.layerID, optParams);
        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

        kernel.workGroups.global = {runInfo.gws0, runInfo.gws1, runInfo.gws2};
        kernel.workGroups.local = {runInfo.lws0, runInfo.lws1, runInfo.lws2};
        kernel.kernelString = GetKernelString(kernelName, jit, entryPoint, params.engineInfo);
        kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, (uint32_t)i});
        kernel.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        ScalarDescriptor s;
        s.t = ScalarDescriptor::Types::UINT32;
        s.v.u32 = lastOffset;
        kernel.scalars.push_back(s);
        kernel.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});

        lastOffset += (uint32_t)input.GetDims()[concatChannelIndex].v;
        effiency = std::max(effiency, runInfo.effiency);
    }

    kd.estimatedTime = effiency;

    return {kd};
}

}  // namespace kernel_selector
