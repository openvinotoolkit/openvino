// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arg_max_min_kernel_topk_radix.h"
#include <kernel_selector_utils.h>
#include <cstdlib>

namespace kernel_selector {

namespace {

constexpr size_t kWgSize = 256;

size_t GetOperationNumber(const arg_max_min_params& params) {
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::FEATURE: return params.outputs[0].Batch().v * params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::Z: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::Y: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::X: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].Y().v;
        default: throw std::invalid_argument("Unsupported axis");
    }
}

std::string GetOperationNumberString(const arg_max_min_params& params) {
    const auto& output = params.outputs[0];
    DimensionAccessHelperJit dims(output);
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return toVectorMulString({dims.x(), dims.y(), dims.z(), dims.f()});
        case ArgMaxMinAxis::FEATURE: return toVectorMulString({dims.x(), dims.y(), dims.z(), dims.b()});
        case ArgMaxMinAxis::Z: return toVectorMulString({dims.y(), dims.z(), dims.f(), dims.b()});
        case ArgMaxMinAxis::Y: return toVectorMulString({dims.x(), dims.z(), dims.f(), dims.b()});
        case ArgMaxMinAxis::X: return toVectorMulString({dims.y(), dims.z(), dims.f(), dims.b()});
        default: throw std::invalid_argument("Unsupported axis");
    }
}

size_t GetSortSize(const arg_max_min_params& params) {
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return params.inputs[0].Batch().v;
        case ArgMaxMinAxis::FEATURE: return params.inputs[0].Feature().v;
        case ArgMaxMinAxis::Z: return params.inputs[0].Z().v;
        case ArgMaxMinAxis::Y: return params.inputs[0].Y().v;
        case ArgMaxMinAxis::X: return params.inputs[0].X().v;
        default: throw std::invalid_argument("Unsupported axis");
    }
}

}  // namespace

ParamsKey ArgMaxMinKernelTopKRadix::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableAllOutputDataType();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::BATCH);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::X);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::Y);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::Z);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::FEATURE);
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    return k;
}

bool ArgMaxMinKernelTopKRadix::Validate(const Params& p) const {
    if (!ArgMaxMinKernelBase::Validate(p))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const auto& params = static_cast<const arg_max_min_params&>(p);

    // Only f16 input for radix approach (bit manipulation)
    if (params.inputs[0].GetDType() != Datatype::F16)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.argMaxMinSortType != ArgMaxMinSortType::VALUE)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const size_t sort_size = GetSortSize(params);

    if (sort_size < 2)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Combined key uses (i & 0xFFFF) as tiebreaker, so N must fit in 16 bits
    if (sort_size > 65535)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // PADDED_K (next power of 2 >= topK) must fit in SLM:
    // sort_keys[PADDED_K] + sort_idxs[PADDED_K] + histogram[256] + scalars
    size_t padded_k = 1;
    while (padded_k < params.topK)
        padded_k <<= 1;
    const size_t slm_needed = padded_k * 2 * sizeof(uint32_t) + 256 * sizeof(uint32_t) + 24;
    if (slm_needed > params.engineInfo.maxLocalMemSize)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    return true;
}

ArgMaxMinKernelBase::DispatchData ArgMaxMinKernelTopKRadix::SetDefault(const arg_max_min_params& params) const {
    DispatchData dispatchData;

    size_t ops_size = 1;
    if (!params.has_dynamic_tensors()) {
        ops_size = GetOperationNumber(params);
    }

    dispatchData.gws = { ops_size * kWgSize, 1, 1 };
    dispatchData.lws = { kWgSize, 1, 1 };

    return dispatchData;
}

void ArgMaxMinKernelTopKRadix::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const arg_max_min_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);

        // Sortable keys buffer: VALUES_NUM * sizeof(uint) per operation
        const size_t sort_size = GetSortSize(prim_params);
        const size_t ops_num = GetOperationNumber(prim_params);
        kd.internalBuffers.clear();
        kd.internalBuffers.push_back(4 * sort_size * ops_num);
    };
}

JitConstants ArgMaxMinKernelTopKRadix::GetJitConstants(const arg_max_min_params& params) const {
    auto jit = ArgMaxMinKernelBase::GetJitConstants(params);

    jit.AddConstant(MakeJitConstant("WG_SIZE", kWgSize));

    // PADDED_K: next power of 2 >= TOP_K (for bitonic sort)
    size_t padded_k = 1;
    while (padded_k < params.topK) padded_k <<= 1;
    jit.AddConstant(MakeJitConstant("PADDED_K", padded_k));

    if (params.has_dynamic_tensors()) {
        jit.AddConstant(MakeJitConstant("OPERATION_NUM", GetOperationNumberString(params)));
    } else {
        jit.AddConstant(MakeJitConstant("OPERATION_NUM", GetOperationNumber(params)));
    }

    if (params.argMaxMinSortType == ArgMaxMinSortType::VALUE)
        jit.AddConstant(MakeJitConstant("SORT_BY_VALUE", 1));

    if (params.values_first)
        jit.AddConstant(MakeJitConstant("TOP_K_ORDER", 1));

    return jit;
}

KernelsData ArgMaxMinKernelTopKRadix::GetKernelsData(const Params& params) const {
    if (!Validate(params))
        return {};

    const auto& orgParams = static_cast<const arg_max_min_params&>(params);
    auto dispatchData = SetDefault(orgParams);
    KernelData kd = KernelData::Default<arg_max_min_params>(params);
    GetUpdateDispatchDataFunc(kd);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT, false, false, 1,
                     GetFusedPrimitiveInputsCount(params), orgParams.outputs_num,
                     orgParams.is_shape_agnostic);

    // Add internal buffer for sortable keys (VALUES_NUM * sizeof(uint) per operation)
    const size_t sort_size = GetSortSize(orgParams);
    const size_t ops_num = GetOperationNumber(orgParams);
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    kd.internalBuffers.push_back(4 * sort_size * ops_num);
    kd.internalBufferDataType = Datatype::UINT32;

    return {kd};
}

KernelsPriority ArgMaxMinKernelTopKRadix::GetKernelsPriority(const Params& p) const {
    const auto& params = static_cast<const arg_max_min_params&>(p);
    const size_t sort_size = GetSortSize(params);

    // Radix sort excels at large sort sizes with large k (e.g., N=8400+, k=300).
    // For k=1 (pure argmax/argmin) or small sort sizes, the axis kernel's
    // simple reduction is more efficient — especially when there are many
    // independent operations that amplify per-WG overhead.
    if (params.topK == 1 || sort_size < 256)
        return FORCE_PRIORITY_5;

    return FORCE_PRIORITY_1;
}

}  // namespace kernel_selector
