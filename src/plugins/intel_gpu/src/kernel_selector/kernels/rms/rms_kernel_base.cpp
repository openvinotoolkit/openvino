// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
bool RMSKernelBase::Validate(const Params& p) const {
    if (!KernelBaseOpenCL::Validate(p))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const rms_params& params = static_cast<const rms_params&>(p);
    auto supported_dyn_layouts = { DataLayout::bfyx, DataLayout::bfzyx };
    if (params.has_dynamic_tensors() && (!layout_is_one_of(params.inputs, supported_dyn_layouts) || !layout_is_one_of(params.outputs, supported_dyn_layouts)))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    return true;
}

JitConstants RMSKernelBase::GetJitConstants(const rms_params& params, RMSKernelBase::DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("EPSILON", params.epsilon));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    return jit;
}

RMSKernelBase::DispatchData RMSKernelBase::SetDefault(const rms_params& params) const {
    DispatchData dispatchData;
    const auto& output = params.outputs[0];

    dispatchData.gws = {output.Batch().v, output.Feature().v, 1};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

void RMSKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const rms_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData RMSKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::RMS);

    if (!Validate(params))
        return {};

    const rms_params& orgParams = static_cast<const rms_params&>(params);
    auto dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<rms_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params);
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
                     2,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     orgParams.is_shape_agnostic);

    return {kd};
}

Datatype RMSKernelBase::GetAccumulatorType(const rms_params& params) const {
    const auto& input_dt = params.inputs[0].GetDType();

    switch (input_dt) {
        case Datatype::F32:
        case Datatype::F16:
            return Datatype::F32;
        case Datatype::INT8: return Datatype::INT32;
        case Datatype::UINT8: return Datatype::INT32;
        default: return Datatype::F32;
    }
}
}  // namespace kernel_selector
