// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kv_cache_kernel_ref.hpp"

#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

void ReorderKVCacheKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = dynamic_cast<const reorder_kv_cache_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = false;
        ScalarDescriptor seq_len;

        seq_len.t = ScalarDescriptor::Types::UINT32;
        seq_len.v.u32 = prim_params.seq_len;
        kd.kernels[0].params.scalars.resize(1);
        kd.kernels[0].params.scalars[0] = seq_len;
    };
}

KernelsData ReorderKVCacheKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    auto kernel_data = KernelData::Default<reorder_kv_cache_params>(params);
    const auto& kernel_params = dynamic_cast<const reorder_kv_cache_params&>(*kernel_data.params);
    const auto dispatch_data = SetDefault(kernel_params);
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params);
    const auto jit_constants = GetJitConstants(kernel_params);
    const auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto& kernel = kernel_data.kernels.front();

    GetUpdateDispatchDataFunc(kernel_data);

    FillCLKernelData(kernel,
                     dispatch_data,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     {},
                     false,
                     false,
                     static_cast<int>(kernel_params.inputs.size()),
                     GetFusedPrimitiveInputsCount(kernel_params),
                     static_cast<int>(kernel_params.outputs.size()),
                     kernel_params.is_shape_agnostic);

    ScalarDescriptor seq_len;
    seq_len.t = ScalarDescriptor::Types::UINT32;
    seq_len.v.u32 = 0;
    kernel.params.scalars.push_back(seq_len);
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});

    return {kernel_data};
}

ParamsKey ReorderKVCacheKernelRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::INT32);
    key.EnableOutputDataType(Datatype::INT32);
    key.EnableInputLayout(DataLayout::bfyx);
    key.EnableOutputLayout(DataLayout::bfyx);
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    key.EnableDynamicShapesSupport();
    return key;
}

bool ReorderKVCacheKernelRef::Validate(const Params& params) const {
    if (params.GetType() != KernelType::REORDER_KV_CACHE) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    const auto& kernel_params = dynamic_cast<const reorder_kv_cache_params&>(params);
    if (kernel_params.inputs.size() != 3) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }
    if (kernel_params.outputs.size() != 1) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    return true;
}

JitConstants ReorderKVCacheKernelRef::GetJitConstants(const reorder_kv_cache_params& kernel_params) const {
    JitConstants jit = MakeBaseParamsJitConstants(kernel_params);
    jit.AddConstant({MakeJitConstant("INDIRECT_AXIS", kernel_params.indirect_axis)});
    return jit;
}

CommonDispatchData ReorderKVCacheKernelRef::SetDefault(const reorder_kv_cache_params& kernel_params) {
    CommonDispatchData dispatch_data;

    auto output = kernel_params.outputs[0];
    dispatch_data.gws = {output.Batch().v * output.Feature().v, Align(output.X().v, 16), 1};
    dispatch_data.lws = {1, 16, 1};

    return dispatch_data;
}

}  // namespace kernel_selector
