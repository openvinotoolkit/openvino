// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_quantize_kernel_opt.h"
#include "kernel_selector_utils.h"
#include <string>


static constexpr size_t simd = 16;

namespace kernel_selector {

enum class DynQuanMode {
    SMALL_GS = 1,
    LARGE_GS = 2,
    PER_TOKEN = 3
};

static std::pair<size_t, size_t> get_input_bf_size(const dynamic_quantize_params& params) {
    size_t input_f = params.inputs[0].Feature().v;
    size_t input_batch = params.inputs[0].Batch().v;
    // 3D input
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        input_f = params.inputs[0].Y().v * params.inputs[0].X().v;
        input_batch = params.inputs[0].Batch().v * params.inputs[0].Feature().v;
    }

    // In Some model, input_f could be dynamic in input0. It refers to IFM value of weight.
    if (params.inputs[0].is_dynamic() && input_f == 0) {
        OPENVINO_ASSERT(params.fc_ifm_size != 0, "[GPU] Invalid fc_ifm_size value");
        input_f = params.fc_ifm_size;
    }

    return {input_batch, input_f};
}

static DynQuanMode get_dynamic_quantize_mode(const dynamic_quantize_params& params) {
    if (params.group_sizes.back() <= 64)
        return DynQuanMode::SMALL_GS;
    else if (params.group_sizes.back() == std::numeric_limits<uint64_t>::max())
        return DynQuanMode::PER_TOKEN;
    else
        return DynQuanMode::LARGE_GS;
}

static size_t get_match_vector_size(const dynamic_quantize_params& params) {
    auto block_sizes = { 8, 4, 2 };
    auto bf = get_input_bf_size(params);
    auto f = bf.second;

    for (auto block_size : block_sizes) {
        if ((f / simd) % block_size == 0) {
            return block_size;
        }
    }

    return 1;
}

ParamsKey DynamicQuantizeKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableDifferentTypes();
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants DynamicQuantizeKernelOpt::GetJitConstants(const dynamic_quantize_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    auto vec_size = get_match_vector_size(params);
    auto bf_size = get_input_bf_size(params);
    auto total_block_num = bf_size.second / (simd * vec_size);
    auto mode = get_dynamic_quantize_mode(params);

    jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));
    jit.AddConstant(MakeJitConstant("SIMD", simd));
    jit.AddConstant(MakeJitConstant("QUANTIZE_GROUP_SIZE", params.group_sizes.back()));
    jit.AddConstant(MakeJitConstant("ASYMMETRIC_QUANTIZATION", params.use_asymmetric_quantization));
    jit.AddConstant(MakeJitConstant("DYNAMIC_QUANTIZAION_IMPL_MODE", static_cast<int>(mode)));
    jit.AddConstant(MakeJitConstant("MODE_SMALL_GS", static_cast<int>(DynQuanMode::SMALL_GS)));
    jit.AddConstant(MakeJitConstant("MODE_LARGE_GS", static_cast<int>(DynQuanMode::LARGE_GS)));
    jit.AddConstant(MakeJitConstant("MODE_PER_TOKEN", static_cast<int>(DynQuanMode::PER_TOKEN)));
    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.outputs[0]));

    if (mode == DynQuanMode::PER_TOKEN)  {
        size_t aligned_block_num = (total_block_num > 32) ? Align(total_block_num, 32) : total_block_num;
        size_t block_num = (total_block_num > 32) ? 32 : total_block_num;
        jit.AddConstant(MakeJitConstant("TOTAL_BLOCK_NUM", total_block_num));
        jit.AddConstant(MakeJitConstant("ALIGNED_BLOCK_NUM", aligned_block_num));
        jit.AddConstant(MakeJitConstant("BLOCK_NUM", block_num));
    }

    return jit;
}

CommonDispatchData DynamicQuantizeKernelOpt::SetDefault(const dynamic_quantize_params& params) const {
    CommonDispatchData dispatchData;
    auto mode = get_dynamic_quantize_mode(params);

    if (mode == DynQuanMode::SMALL_GS) {
        auto bf_size = get_input_bf_size(params);
        dispatchData.gws = {bf_size.first, bf_size.second / params.group_sizes.back(), 1};
        dispatchData.lws = {1, 1, 1};
    } else if (mode == DynQuanMode::LARGE_GS) {
        auto vec_size = get_match_vector_size(params);
        auto bf_size = get_input_bf_size(params);
        size_t dyn_quan_gs = params.group_sizes.back() == UINT64_MAX ? bf_size.second : params.group_sizes.back();
        size_t total_block_num = bf_size.second / (simd * vec_size);
        size_t batch = bf_size.first;

        dispatchData.gws = {simd, total_block_num, batch};
        // NOTE: this implementation is not directly applicable to per-token case because dyn_quan_gs / (simd*vec_size) may exceed LWS size limit.
        dispatchData.lws = {simd, dyn_quan_gs / (simd * vec_size), 1};
    } else if (mode == DynQuanMode::PER_TOKEN) {
        auto vec_size = get_match_vector_size(params);
        auto bf_size = get_input_bf_size(params);
        size_t total_block_num = bf_size.second / (simd * vec_size);
        size_t batch = bf_size.first;
        size_t block_num = (total_block_num > 32) ? 32 : total_block_num;

        dispatchData.gws = {simd, block_num, batch};
        dispatchData.lws = {simd, block_num, 1};
    } else {
        OPENVINO_ASSERT(false);
    }

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

        GPU_DEBUG_TRACE_DETAIL << "Update Dispatch data DynamicQuantizeKernelOpt gws : " << dispatchData.gws[0] << ", "
                << dispatchData.gws[1] << ", " << dispatchData.gws[2] << std::endl;
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

    return {kd};
}

KernelsPriority DynamicQuantizeKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

bool DynamicQuantizeKernelOpt::Validate(const Params& params) const {
    if (!KernelBaseOpenCL::Validate(params))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    const auto& dq_params = static_cast<const dynamic_quantize_params&>(params);


    auto bf = get_input_bf_size(dq_params);
    if (((bf.second) % (simd * 2)) != 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (dq_params.inputs[0].GetPaddedVal() != 0 || dq_params.outputs[0].GetPaddedVal() != 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (dq_params.append_axis != -1)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    for (size_t i = 0; i < dq_params.group_sizes.size() - 1; i++) {
        if (dq_params.group_sizes[i] != 1)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // Allow only default scales order
    const auto& scales_output_order = dq_params.scales_output_order;
    if (!scales_output_order.empty()) {
        for (size_t i = 0; i < scales_output_order.size(); i++)
            if (scales_output_order[i] != i)
                DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    if (dq_params.use_asymmetric_quantization) {
        if (dq_params.combine_scales_and_zp)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        if (dq_params.outputs[0].GetDType() != Datatype::UINT8)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    return true;
}


}  // namespace kernel_selector

