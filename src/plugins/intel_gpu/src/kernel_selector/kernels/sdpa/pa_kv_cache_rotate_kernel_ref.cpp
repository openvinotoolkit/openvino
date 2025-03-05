// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_kv_cache_rotate_kernel_ref.h"
#include "sdpa_kernel_base.h"

#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

constexpr size_t subgroup_size = 16;
constexpr size_t paged_attention_block_size = 16;

KernelsData KVCacheRotateKernelRef::GetKernelsData(const Params& p) const {
    if (!Validate(p)) {
        return {};
    }

    KernelData kd = KernelData::Default<kv_cache_rotate_params>(p);
    kd.needs_sub_kernels_sync = false;
    GetUpdateDispatchDataFunc(kd);

    const auto& params = static_cast<const kv_cache_rotate_params&>(p);
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

    return {kd};
}

ParamsKey KVCacheRotateKernelRef::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
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

bool KVCacheRotateKernelRef::Validate(const Params& params) const {
    if (params.GetType() != KernelType::PA_KV_CACHE_ROTATE)
        return false;

    const auto& kernel_params = dynamic_cast<const kv_cache_rotate_params&>(params);
    if (kernel_params.inputs.size() != 3)
        return false;

    if (kernel_params.outputs.size() != 1)
        return false;

    if (!kernel_params.conf.is_paged_attention)
        return false;

    if (kernel_params.conf.paged_attention_block_size != static_cast<int64_t>(paged_attention_block_size))
        return false;

    return true;
}

JitConstants KVCacheRotateKernelRef::GetJitConstants(const kv_cache_rotate_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("HEAD_SIZE", params.conf.head_size));
    jit.AddConstant(MakeJitConstant("HEADS_NUM", params.conf.heads_num));
    jit.AddConstant(MakeJitConstant("KV_HEADS_NUM", params.conf.kv_heads_num));
    jit.AddConstant(MakeJitConstant("PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size));
    jit.AddConstant(MakeJitConstant("SUBGROUP_SIZE", subgroup_size));
    jit.AddConstant(MakeJitConstant("IS_KV_COMPRESSED", params.conf.is_kv_compressed));
    jit.Merge(MakeTypeJitConstants(params.original_cache_dt, "UNCOMPRESSED"));

    if (params.conf.is_kv_compressed) {
        auto scales_zp_size = BytesPerElement(params.original_cache_dt) * 2; // scale + zp
        jit.AddConstant(MakeJitConstant("SCALE_ZP_SIZE_PER_TOKEN", scales_zp_size));
        jit.AddConstant(MakeJitConstant("ADJUSTED_HEAD_SIZE", params.conf.head_size + scales_zp_size));
    } else {
        jit.AddConstant(MakeJitConstant("ADJUSTED_HEAD_SIZE", params.conf.head_size));
    }

    return jit;
}

CommonDispatchData KVCacheRotateKernelRef::SetDefault(const kv_cache_rotate_params& params) {
    CommonDispatchData dispatch_data;

    const auto& rotated_block_indices_input = params.inputs[0];
    if (!rotated_block_indices_input.is_dynamic()) {
        auto heads_number = static_cast<size_t>(params.conf.kv_heads_num);
        auto blocks_to_rotate = static_cast<size_t>(rotated_block_indices_input.Batch().v);

        dispatch_data.gws = { subgroup_size,
                              heads_number,
                              blocks_to_rotate };
        dispatch_data.lws = { subgroup_size,
                              params.conf.is_kv_compressed ? 1 : heads_number,
                              1 };
    }

    return dispatch_data;
}

void KVCacheRotateKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const kv_cache_rotate_params&>(params);
        const auto& rotated_block_indices_input = prim_params.inputs[0];

        auto dispatch_data = SetDefault(prim_params);

        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatch_data.gws;
        kd.kernels[0].params.workGroups.local = dispatch_data.lws;
        kd.kernels[0].skip_execution = rotated_block_indices_input.Batch().v == 0;
    };
}

}  // namespace kernel_selector
