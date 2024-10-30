// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_quantize_kernel_opt.h"
#include "kernel_selector_utils.h"
#include <string>


static constexpr size_t simd = 16;

namespace kernel_selector {
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

static size_t get_match_vector_size(const dynamic_quantize_params& params) {
    auto block_sizes = { 8, 4, 2 };

    for (auto block_size : block_sizes) {
        if (((params.inputs[0].X().v * params.inputs[0].Y().v) / simd) % block_size == 0) {
            return block_size;
        }
    }

    return 1;
}

ParamsKey DynamicQuantizeKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
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

    auto vec_size = get_match_vector_size(params);
    auto bf_size = get_input_bf_size(params);
    auto total_block_num = bf_size.second / (simd * vec_size);
    size_t aligned_block_num = (total_block_num > 32) ? Align(total_block_num, 32) : total_block_num;
    size_t block_num = (total_block_num > 32) ? 32 : total_block_num;

    jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));
    jit.AddConstant(MakeJitConstant("SIMD", simd));
    jit.AddConstant(MakeJitConstant("TOTAL_BLOCK_NUM", total_block_num));
    jit.AddConstant(MakeJitConstant("ALIGNED_BLOCK_NUM", aligned_block_num));
    jit.AddConstant(MakeJitConstant("BLOCK_NUM", block_num));
    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.outputs[0]));

    return jit;
}

CommonDispatchData DynamicQuantizeKernelOpt::SetDefault(const dynamic_quantize_params& params) const {
    CommonDispatchData dispatchData;

    auto vec_size = get_match_vector_size(params);
    auto bf_size = get_input_bf_size(params);
    size_t total_block_num = bf_size.second / (simd * vec_size);
    size_t batch = get_input_bf_size(params).first;
    size_t block_num = (total_block_num > 32) ? 32 : total_block_num;

    dispatchData.gws = {simd, block_num, batch};
    dispatchData.lws = {simd, block_num, 1};

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
        return false;

    const auto& dq_params = static_cast<const dynamic_quantize_params&>(params);

    // Todo : Add proper exception here
    if (((dq_params.inputs[0].X().v * dq_params.inputs[0].Y().v) % (simd * 2)) != 0)
        return false;

    if (dq_params.inputs[0].GetPaddedVal() != 0 || dq_params.outputs[0].GetPaddedVal() != 0)
        return false;

    if (dq_params.append_axis != -1)
        return false;

    if (dq_params.group_sizes.back() != UINT64_MAX)
        return false;

    if (!dq_params.scales_output_order.empty())
        return false;

    return true;
}
}  // namespace kernel_selector

