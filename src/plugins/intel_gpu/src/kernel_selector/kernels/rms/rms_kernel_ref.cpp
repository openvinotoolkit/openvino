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
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants RMSKernelRef::GetJitConstants(const rms_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("EPSILON", params.epsilon)});

    return jit;
}

CommonDispatchData RMSKernelRef::SetDefault(const rms_params& params) const {
    CommonDispatchData dispatchData;

    dispatchData.gws = {1, 1, 1};
    dispatchData.lws = dispatchData.gws;

    return dispatchData;
}

KernelsData RMSKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::RMS);

    if (!Validate(params, options))
        return {};

    const rms_params& orgParams = static_cast<const rms_params&>(params);
    auto dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<rms_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const rms_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };

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

bool RMSKernelRef::Validate(const Params& params, const optional_params& options) const {
    if (!KernelBaseOpenCL::Validate(params, options))
        return false;

    return true;
}
}  // namespace kernel_selector
