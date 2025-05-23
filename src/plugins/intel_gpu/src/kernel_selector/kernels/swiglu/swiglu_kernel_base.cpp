// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swiglu_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
CommonDispatchData SwiGLUKernelBase::SetDefault(const swiglu_params& params) const {
    CommonDispatchData dispatchData;

    dispatchData.gws = GetTensorFriendlyWorkGroups(params.outputs[0]);
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

JitConstants SwiGLUKernelBase::GetJitConstants(const swiglu_params& params, const CommonDispatchData& dispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("AXIS", params.axis)});
    jit.AddConstants({MakeJitConstant("SPLIT_LENGTH", params.split_length)});
    jit.AddConstants({MakeJitConstant("GLU_TYPE", params.glu_type)});
    jit.AddConstants({MakeJitConstant("LWS0", dispatchData.lws[0])});
    jit.AddConstants({MakeJitConstant("LWS1", dispatchData.lws[1])});
    jit.AddConstants({MakeJitConstant("LWS2", dispatchData.lws[2])});
    const std::string type_suffix = (GetAccumulatorType(params) == Datatype::F32) ? "f" : "h";
    if (params.glu_type == ov::op::internal::GLU::GluType::Gelu) {
        jit.AddConstants({MakeJitConstant("GEGLU_HALF", "0.5" + type_suffix)});
        jit.AddConstants({MakeJitConstant("GEGLU_MULT", "0.7071067811865475" + type_suffix)});
    } else if (params.glu_type == ov::op::internal::GLU::GluType::Gelu_Tanh) {
        jit.AddConstants({MakeJitConstant("GEGLU_HALF", "0.5" + type_suffix)});
        jit.AddConstants({MakeJitConstant("GEGLU_MULT", "0.044715" + type_suffix)});
        jit.AddConstants({MakeJitConstant("GEGLU_SQUARE_2_OVER_PI", "0.79788458347320556640625" + type_suffix)});
    }
    jit.AddConstants({MakeJitConstant("SPLIT_TO_GLU_IDX", params.split_to_glu_idx)});
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.outputs[0]));

    return jit;
}

KernelsData SwiGLUKernelBase::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::SWIGLU);

    if (!Validate(params))
        return {};

    const swiglu_params& orgParams = static_cast<const swiglu_params&>(params);
    auto dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<swiglu_params>(params);

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
                     1,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     orgParams.is_shape_agnostic);

    return {kd};
}


bool SwiGLUKernelBase::Validate(const Params& params) const {
    if (!KernelBaseOpenCL::Validate(params))
        return false;

    return true;
}

Datatype SwiGLUKernelBase::GetAccumulatorType(const swiglu_params& params) const {
    Datatype types[] = { Datatype::F32, Datatype::F16, Datatype::INT64, Datatype::INT32, Datatype::UINT32};

    for (Datatype type : types)
        for (auto& in : params.inputs)
            if (in.GetDType() == type)
                return type;

    return Datatype::F32;
}

void SwiGLUKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const swiglu_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}
}  // namespace kernel_selector
