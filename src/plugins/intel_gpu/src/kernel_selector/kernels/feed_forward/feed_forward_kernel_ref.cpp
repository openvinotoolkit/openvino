// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "feed_forward_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey FeedForwardKernelRef::GetSupportedKey() const {
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
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants FeedForwardKernelRef::GetJitConstants(const feed_forward_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    auto acc_dt = params.inputs[0].GetDType();
    jit.Merge(MakeTypeJitConstants(acc_dt, "ACCUMULATOR"));
    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.outputs[0]));

    const auto& prim_params = dynamic_cast<const feed_forward_params&>(params);
    auto is_scalar = [&]() {
        for (size_t i = 1; i < prim_params.inputs.size(); ++i) {
            if (prim_params.inputs[i].LogicalSize() != 1)
                return false;
        }
        return true;
    };
    jit.AddConstants({MakeJitConstant("IS_SCALAR", is_scalar())});

    return jit;
}

CommonDispatchData FeedForwardKernelRef::SetDefault(const feed_forward_params& params) const {
    CommonDispatchData dispatchData;

    dispatchData.gws = GetTensorFriendlyWorkGroups(params.outputs[0]);
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData FeedForwardKernelRef::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::FEED_FORWARD);
    const auto& prim_params = dynamic_cast<const feed_forward_params&>(params);

    if (!Validate(params))
        return {};

    const feed_forward_params& orgParams = static_cast<const feed_forward_params&>(params);
    auto dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<feed_forward_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
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
                     static_cast<int>(prim_params.inputs.size()),
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     orgParams.has_dynamic_tensors());

    return {kd};
}

KernelsPriority FeedForwardKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool FeedForwardKernelRef::Validate(const Params& params) const {
    if (!KernelBaseOpenCL::Validate(params))
        return false;

    return true;
}

void FeedForwardKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const feed_forward_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

}  // namespace kernel_selector
