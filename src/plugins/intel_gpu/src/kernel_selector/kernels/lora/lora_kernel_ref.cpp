// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey LoRAKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();

    return k;
}

CommonDispatchData LoRAKernelRef::SetDefault(const lora_params& params) const {
    CommonDispatchData dispatchData;

    const auto& output = params.outputs[0];

    const auto& lora_rank_dim = params.inputs[3].Feature();
    size_t lora_rank = lora_rank_dim.is_dynamic ? 1 : lora_rank_dim.v;

    dispatchData.gws = { output.Batch().v * output.Feature().v,
                         Align(output.Y().v * output.X().v, lora_rank),
                         1};

    dispatchData.lws = { 1, lora_rank, 1 };

    return dispatchData;
}

JitConstants LoRAKernelRef::GetJitConstants(const lora_params& params) const {
    auto jit = Parent::GetJitConstants(params);

    jit.AddConstant(MakeJitConstant("MAX_LORA_RANK", 256));

    return jit;
}

void LoRAKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const lora_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData LoRAKernelRef::GetKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<lora_params>(params);
    const auto& prim_params = dynamic_cast<const lora_params&>(params);

    if (!Validate(params)) {
        return {};
    }

    auto dispatchData = SetDefault(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto cldnn_jit = GetJitConstants(prim_params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false, static_cast<int>(prim_params.inputs.size()),
                     GetFusedPrimitiveInputsCount(params), 1, prim_params.is_shape_agnostic);
    return { kd };
}

KernelsPriority LoRAKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

}  // namespace kernel_selector
