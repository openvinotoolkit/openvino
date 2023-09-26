// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey RMSKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    // k.EnableDynamicShapesSupport();
    return k;
}

JitConstants RMSKernelRef::GetJitConstants(const rms_params& params, RMSKernelRef::DispatchData DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("EPSILON", params.epsilon),
        MakeJitConstant("DATA_SIZE", DispatchData.dataSize),
        MakeJitConstant("DATA_COUNT", DispatchData.dataCount),
        MakeJitConstant("SLM_SIZE", DispatchData.slmSize),
        MakeJitConstant("VEC_SIZE", 8),
        MakeJitConstant("VLOAD", "CAT(vload, VEC_SIZE)"),
        MakeJitConstant("VSTORE", "CAT(vstore, VEC_SIZE)"),
        MakeJitConstant("INPUTVTYPE", "CAT(INPUT0_TYPE, VEC_SIZE)"),
        MakeJitConstant("OUTPUTVTYPE", "CAT(OUTPUT_TYPE, VEC_SIZE)"),
        MakeJitConstant("AS_INPUTVTYPE", "CAT(as_, INPUTVTYPE)"),
        MakeJitConstant("AS_OUTPUTVTYPE", "CAT(as_, OUTPUTVTYPE)")
    });
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    return jit;
}

RMSKernelRef::DispatchData RMSKernelRef::SetDefault(const rms_params& params) const {
    DispatchData dispatchData;
    const auto& input = params.inputs[0];

    dispatchData.dataSize = input.X().v * input.Y().v * input.Z().v;
    dispatchData.dataCount = input.Batch().v * input.Feature().v;
    dispatchData.slmSize = dispatchData.dataSize / 8;

    dispatchData.gws[0] = dispatchData.slmSize;
    dispatchData.gws[1] = dispatchData.dataCount;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = dispatchData.slmSize;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsData RMSKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::RMS);

    if (!Validate(params, options))
        return {};

    const rms_params& orgParams = static_cast<const rms_params&>(params);
    auto dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<rms_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    // kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
    //     const auto& prim_params = static_cast<const rms_params&>(params);
    //     auto dispatchData = SetDefault(prim_params);
    //     OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
    //     kd.kernels[0].params.workGroups.global = dispatchData.gws;
    //     kd.kernels[0].params.workGroups.local = dispatchData.lws;
    //     kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    // };

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
                     orgParams.outputs[0].is_dynamic());

    return {kd};
}

KernelsPriority RMSKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

Datatype RMSKernelRef::GetAccumulatorType(const rms_params& params) const {
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

bool RMSKernelRef::Validate(const Params& params, const optional_params& options) const {
    if (!KernelBaseOpenCL::Validate(params, options))
        return false;

    return true;
}
}  // namespace kernel_selector
