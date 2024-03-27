// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_kv_cache_update_kernel_ref.h"
#include "sdpa_kernel_base.h"

#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

constexpr size_t SUBGROUP_SIZE = 16;
constexpr size_t VLLM_BLOCK_SIZE = 16;

void KVCacheUpdateKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& kv_cache_params = static_cast<const kv_cache_update_params&>(params);

        GPU_DEBUG_TRACE_DETAIL << "key data shape : " << kv_cache_params.inputs[0].Batch().v  << " " << kv_cache_params.inputs[0].Feature().v << "\n";
        GPU_DEBUG_TRACE_DETAIL << "value data shape : " << kv_cache_params.inputs[1].Batch().v  << " " << kv_cache_params.inputs[1].Feature().v << "\n";
        GPU_DEBUG_TRACE_DETAIL << "subsequence_begins data shape : " << kv_cache_params.inputs[2].Batch().v  << " " << kv_cache_params.inputs[2].Feature().v << "\n";
        GPU_DEBUG_TRACE_DETAIL << "block_indices shape : " << kv_cache_params.inputs[3].Batch().v  << " " << kv_cache_params.inputs[3].Feature().v << "\n";

        const auto& prim_params = dynamic_cast<const kv_cache_update_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = false;
    };
}

KernelsData KVCacheUpdateKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<kv_cache_update_params>(params);
    kd.needs_sub_kernels_sync = false;
    GetUpdateDispatchDataFunc(kd);

    const auto& kernel_params = static_cast<const kv_cache_update_params&>(params);
    const auto dispatch_data = SetDefault(kernel_params);
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params);
    const auto jit_constants = GetJitConstants(kernel_params);
    const auto jit = CreateJit(kernelName, jit_constants, entry_point);

    auto& kernel = kd.kernels[0];
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

    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5});

    return {kd};
}

ParamsKey KVCacheUpdateKernelRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::F16);
    key.EnableInputDataType(Datatype::F32);
    key.EnableInputDataType(Datatype::INT32);
    key.EnableOutputDataType(Datatype::F16);
    key.EnableOutputDataType(Datatype::F32);
    key.EnableOutputDataType(Datatype::INT32);
    key.EnableInputLayout(DataLayout::bfyx);
    key.EnableOutputLayout(DataLayout::bfyx);
    key.EnableOutputLayout(DataLayout::bfzyx);
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    key.EnableDynamicShapesSupport();
    key.EnableDifferentTypes();
    return key;
}

bool KVCacheUpdateKernelRef::Validate(const Params& params) const {
    if (params.GetType() != KernelType::PA_KV_CACHE_UPDATE) {
        return false;
    }

    const auto& kernel_params = dynamic_cast<const kv_cache_update_params&>(params);
    if (kernel_params.inputs.size() != 6) {
        return false;
    }

    if (kernel_params.outputs.size() != 2) {
        return false;
    }

    return true;
}

JitConstants KVCacheUpdateKernelRef::GetJitConstants(const kv_cache_update_params& kernel_params) const {
    JitConstants jit = MakeBaseParamsJitConstants(kernel_params);

    jit.AddConstant(MakeJitConstant("SUBGROUP_SIZE", SUBGROUP_SIZE));
    jit.AddConstant(MakeJitConstant("VLLM_BLOCK_SIZE", VLLM_BLOCK_SIZE));
    jit.AddConstant(MakeJitConstant("NUM_HEADS", kernel_params.conf.heads_num));
    jit.AddConstant(MakeJitConstant("HEAD_SIZE", kernel_params.conf.head_size));

    return jit;
}

CommonDispatchData KVCacheUpdateKernelRef::SetDefault(const kv_cache_update_params& kernel_params) {
    CommonDispatchData dispatch_data;

    const auto& key_cache = kernel_params.outputs[0];
    const auto& value_cache = kernel_params.outputs[1];
    if (!value_cache.is_dynamic() && !key_cache.is_dynamic()) {
        bool is_prefill = kernel_params.inputs[0].Batch().v != kernel_params.inputs[4].Batch().v;

        if (is_prefill) {
            const auto& block_indices_input = kernel_params.inputs[3];

            auto blocks_number = block_indices_input.Batch().v;
            auto heads_number = static_cast<size_t>(kernel_params.conf.heads_num);

            dispatch_data.gws = {blocks_number, heads_number, SUBGROUP_SIZE};
            dispatch_data.lws = {1, 1, SUBGROUP_SIZE};
        } else {
            const auto& key_input = kernel_params.inputs[0];

            auto tokens_number = key_input.Batch().v;
            auto heads_number = static_cast<size_t>(kernel_params.conf.heads_num);

            dispatch_data.gws = {tokens_number, heads_number, SUBGROUP_SIZE};
            dispatch_data.lws = {1, 1, SUBGROUP_SIZE};
        }
    }

    return dispatch_data;
}

}  // namespace kernel_selector
