// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_quantize_kernel_opt.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey DynamicQuantizeKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants DynamicQuantizeKernelOpt::GetJitConstants(const dynamic_quantize_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.outputs[0]));

    return jit;
}

CommonDispatchData DynamicQuantizeKernelOpt::SetDefault(const dynamic_quantize_params& params) const {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    CommonDispatchData dispatchData;

    dispatchData.gws = {64, 1, params.inputs[0].Batch().v * params.inputs[0].Feature().v};
    dispatchData.lws = {64, 1, 1};

    return dispatchData;
}

void DynamicQuantizeKernelOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const dynamic_quantize_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = false;
    };
}

KernelsData DynamicQuantizeKernelOpt::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::DYNAMIC_QUANTIZE);

    if (!Validate(params))
        return {};

    const dynamic_quantize_params& prim_params = static_cast<const dynamic_quantize_params&>(params);
    auto dispatchData = SetDefault(prim_params);

    KernelData kd = KernelData::Default<dynamic_quantize_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
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
                     static_cast<int>(prim_params.outputs.size()),
                     prim_params.is_shape_agnostic);

    // std::cout << ">> Select dynamic_quantize_kernel_opt : " << prim_params.outputs.size() << std::endl;

    return {kd};
}

KernelsPriority DynamicQuantizeKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

Datatype DynamicQuantizeKernelOpt::GetAccumulatorType(const dynamic_quantize_params& params) const {
    Datatype types[] = { Datatype::F32, Datatype::F16, Datatype::INT64, Datatype::INT32, Datatype::UINT32};

    for (Datatype type : types)
        for (auto& in : params.inputs)
            if (in.GetDType() == type)
                return type;
    return Datatype::F32;
}

bool DynamicQuantizeKernelOpt::Validate(const Params& params) const {
    if (!KernelBaseOpenCL::Validate(params))
        return false;

    const auto& dq_params = static_cast<const dynamic_quantize_params&>(params);

    // Todo : Add proper exception here
    if ((dq_params.outputs[0].X().v * dq_params.outputs[0].Y().v % 32) != 0)
        return false;

    return true;
}
}  // namespace kernel_selector

