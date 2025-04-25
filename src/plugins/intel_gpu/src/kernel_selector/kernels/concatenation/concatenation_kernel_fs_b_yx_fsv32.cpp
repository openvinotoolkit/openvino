// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

DeviceFeaturesKey ConcatenationKernel_fs_b_yx_fsv32::get_required_device_features_key(const Params& params) const {
    return get_common_subgroups_device_features_key(params);
}

bool ConcatenationKernel_fs_b_yx_fsv32::Validate(const Params& p) const {
    if (!ConcatenationKernelBase::Validate(p)) {
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
    DispatchData dispatchData = ConcatenationKernelBase::SetDefault(params);
    const auto& input = params.inputs[0];

    dispatchData.gws[0] = input.X().v;
    dispatchData.gws[1] = input.Y().v;
    dispatchData.gws[2] = CeilDiv(input.Feature().v, fsv) * subGroupSize * input.Batch().v;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = subGroupSize;

    return dispatchData;
}

KernelsPriority ConcatenationKernel_fs_b_yx_fsv32::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

JitConstants ConcatenationKernel_fs_b_yx_fsv32::GetJitConstants(const concatenation_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("ALIGNED", params.isAligned));
    jit.AddConstant(MakeJitConstant("FSV", fsv));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subGroupSize));
    jit.AddConstant(MakeJitConstant("FSV_PER_THREAD", fsvPerThread));

    return jit;
}

KernelsData ConcatenationKernel_fs_b_yx_fsv32::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const concatenation_params& orgParams = static_cast<const concatenation_params&>(params);

    KernelData kd = KernelData::Default<concatenation_params>(params, orgParams.inputs.size());

    uint32_t lastOffset = 0;
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
        DispatchData dispatchData = SetDefault(newParams);
        auto cldnnJit = GetJitConstants(newParams);
        auto entryPoint = GetEntryPoint(kernelName, newParams.layerID, params,  i);
        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

        kernel.code.kernelString = GetKernelString(kernelName, jit, entryPoint, params.engineInfo);
        kernel.params.workGroups.global = dispatchData.gws;
        kernel.params.workGroups.local = dispatchData.lws;
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, (uint32_t)i});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        kernel.skip_execution = KernelData::SkipKernelExecution(newParams);

        ScalarDescriptor s;
        s.t = ScalarDescriptor::Types::UINT32;
        s.v.u32 = lastOffset;
        kernel.params.scalars.push_back(s);
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});

        auto concatChannelIndex = DataTensor::Channelndex(orgParams.inputs[i].GetLayout(), GetConcatChannel(orgParams));
        OPENVINO_ASSERT(concatChannelIndex >= 0, "concatChannelIndex shouldn't be negative");
        lastOffset += (uint32_t)input.GetDims()[concatChannelIndex].v;
    }

    return {kd};
}

}  // namespace kernel_selector
