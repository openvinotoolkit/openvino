// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
bool RoPEKernelBase::Validate(const Params& p, const optional_params& o) const {
    return KernelBaseOpenCL::Validate(p, o);
}

JitConstants RoPEKernelBase::GetJitConstants(const rope_params& params, RoPEKernelBase::DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("HEAD_SIZE", params.head_size));
    jit.AddConstant(MakeJitConstant("HALF_ROTARY_NDIMS", params.rotary_ndims / 2));

    return jit;
}

RoPEKernelBase::DispatchData RoPEKernelBase::SetDefault(const rope_params& params) const {
    DispatchData dispatchData;
    const auto& input = params.inputs[0];

    dispatchData.gws = {input.Batch().v, input.Feature().v, params.head_cnt * (params.rotary_ndims / 2)};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

void RoPEKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const rope_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData RoPEKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::ROPE);

    if (!Validate(params, options))
        return {};

    const rope_params& orgParams = static_cast<const rope_params&>(params);
    auto dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<rope_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     2, // TODO: Change num of inputs
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     orgParams.outputs[0].is_dynamic());

    return {kd};
}

}  // namespace kernel_selector
