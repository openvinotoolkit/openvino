// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_kv_cache_update_kernel_ref.h"
#include "sdpa_kernel_base.h"

#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

constexpr size_t subgroup_size = 16;
constexpr size_t paged_attention_block_size = 16;

static size_t get_generate_stage_block_size(size_t head_size) {
    auto preferred_block_size = { 4, 2, 1 };
    for (const auto& block_size : preferred_block_size) {
        if (head_size % block_size == 0) {
            return block_size;
        }
    }

    return 1;
}

KernelsData KVCacheUpdateKernelRef::GetKernelsData(const Params& p) const {
    if (!Validate(p)) {
        return {};
    }

    KernelData kd = KernelData::Default<kv_cache_update_params>(p);
    kd.needs_sub_kernels_sync = false;
    GetUpdateDispatchDataFunc(kd);

    const auto& params = static_cast<const kv_cache_update_params&>(p);
    const auto dispatch_data = SetDefault(params);
    const auto entry_point = GetEntryPoint(kernelName, params.layerID, p);
    const auto jit_constants = GetJitConstants(params);
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
                     static_cast<int>(params.inputs.size()),
                     GetFusedPrimitiveInputsCount(params),
                     static_cast<int>(params.outputs.size()),
                     params.is_shape_agnostic);

    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});

    return {kd};
}

ParamsKey KVCacheUpdateKernelRef::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();

    return k;
}

bool KVCacheUpdateKernelRef::Validate(const Params& params) const {
    if (params.GetType() != KernelType::PA_KV_CACHE_UPDATE)
        return false;

    const auto& kernel_params = dynamic_cast<const kv_cache_update_params&>(params);
    if (kernel_params.inputs.size() != 5)
        return false;

    if (kernel_params.outputs.size() != 2)
        return false;

    if (!kernel_params.conf.is_paged_attention)
        return false;

    if (kernel_params.conf.paged_attention_block_size != static_cast<int64_t>(paged_attention_block_size))
        return false;

    return true;
}

JitConstants KVCacheUpdateKernelRef::GetJitConstants(const kv_cache_update_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("HEAD_SIZE", params.conf.head_size));
    jit.AddConstant(MakeJitConstant("HEADS_NUM", params.conf.heads_num));
    jit.AddConstant(MakeJitConstant("KV_HEADS_NUM", params.conf.kv_heads_num));
    jit.AddConstant(MakeJitConstant("PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size));
    jit.AddConstant(MakeJitConstant("SUBGROUP_SIZE", subgroup_size));
    jit.AddConstant(MakeJitConstant("GENERATE_STAGE_BLOCK_SIZE", get_generate_stage_block_size(params.conf.head_size)));

    return jit;
}

CommonDispatchData KVCacheUpdateKernelRef::SetDefault(const kv_cache_update_params& params) {
    CommonDispatchData dispatch_data;

    const auto& key_cache = params.outputs[0];
    const auto& value_cache = params.outputs[1];
    if (!value_cache.is_dynamic() && !key_cache.is_dynamic()) {
        bool is_prefill = params.inputs[0].Batch().v != params.inputs[2].Batch().v;
        auto heads_number = static_cast<size_t>(params.conf.kv_heads_num);

        if (is_prefill) {
            const auto& block_indices_input = params.inputs[3];
            const auto blocks_number = block_indices_input.Batch().v;

            dispatch_data.gws = { blocks_number,
                                  heads_number,
                                  subgroup_size};
            dispatch_data.lws = { 1, 1, subgroup_size };
        } else {
            const auto& key_input = params.inputs[0];
            const auto sequences_number = key_input.Batch().v;

            dispatch_data.gws = { sequences_number,
                                  heads_number,
                                  subgroup_size };
            dispatch_data.lws = { 1, 1, subgroup_size };
        }
    }

    return dispatch_data;
}

void KVCacheUpdateKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const kv_cache_update_params&>(params);

        auto dispatch_data = SetDefault(prim_params);

        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatch_data.gws;
        kd.kernels[0].params.workGroups.local = dispatch_data.lws;
        kd.kernels[0].skip_execution = false;
    };
}

}  // namespace kernel_selector
