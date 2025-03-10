// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <kernel_selector_utils.h>
#include "arg_max_min_kernel_axis.h"

namespace kernel_selector {

namespace {
size_t getOperationNumber(const arg_max_min_params& params) {
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::FEATURE: return params.outputs[0].Batch().v * params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::Z: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::Y: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::X: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].Y().v;
        default:
            throw std::invalid_argument("Unsupported axis");
    }
}

std::string getOperationNumberString(const arg_max_min_params& params) {
    const auto& output = params.outputs[0];
    DimensionAccessHelperJit dims(output);
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return toVectorMulString({dims.x(), dims.y(), dims.z(), dims.f()});
        case ArgMaxMinAxis::FEATURE: return toVectorMulString({dims.x(), dims.y(), dims.z(), dims.b()});
        case ArgMaxMinAxis::Z: return toVectorMulString({dims.y(), dims.z(), dims.f(), dims.b()});
        case ArgMaxMinAxis::Y: return toVectorMulString({dims.x(), dims.z(), dims.f(), dims.b()});
        case ArgMaxMinAxis::X: return toVectorMulString({dims.y(), dims.z(), dims.f(), dims.b()});
        default:
            throw std::invalid_argument("Unsupported axis");
    }
}

size_t getSortSize(const arg_max_min_params& params) {
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return params.inputs[0].Batch().v;
        case ArgMaxMinAxis::FEATURE: return params.inputs[0].Feature().v;
        case ArgMaxMinAxis::Z: return params.inputs[0].Z().v;
        case ArgMaxMinAxis::Y: return params.inputs[0].Y().v;
        case ArgMaxMinAxis::X: return params.inputs[0].X().v;
        default:
            throw std::invalid_argument("Unsupported axis");
    }
}
}  // namespace

ParamsKey ArgMaxMinKernelAxis::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableAllOutputDataType();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::BATCH);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::X);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::Y);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::Z);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::FEATURE);
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    k.EnableDynamicShapesSupport();
    return k;
}

bool ArgMaxMinKernelAxis::Validate(const Params& p) const {
    if (!ArgMaxMinKernelBase::Validate(p)) {
        return false;
    }

    const arg_max_min_params& params = static_cast<const arg_max_min_params&>(p);

    if (params.inputs.size() > 1) {
        if (params.inputs[1].PitchesDifferFromLogicalDims() || params.outputs[0].PitchesDifferFromLogicalDims())
            return false;
    }

    return true;
}

ArgMaxMinKernelBase::DispatchData ArgMaxMinKernelAxis::SetDefault(const arg_max_min_params& params) const {
    DispatchData dispatchData;

    if (!params.has_dynamic_tensors()) {
        size_t ops_size = getOperationNumber(params);
        ops_size = ops_size > 1 ? Align(ops_size, 32) : 1;
        size_t sort_size = params.argMaxMinSortType == ArgMaxMinSortType::VALUE ? getSortSize(params) : 1;

        dispatchData.gws = { ops_size, sort_size, 1 };
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    }

    return dispatchData;
}

void ArgMaxMinKernelAxis::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const arg_max_min_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);

        const size_t elem_size = prim_params.inputs[0].ElementSize();
        const size_t iav_type_size = elem_size + 4;
        const size_t sort_size = getSortSize(prim_params);
        const size_t ops_size = getOperationNumber(prim_params);
        const size_t group_size = prim_params.topK >= 8 ? prim_params.topK : 8;
        const size_t group_num = ((sort_size - 1) / group_size) + 1;

        kd.internalBuffers.clear();
        kd.internalBuffers.push_back(iav_type_size * sort_size * ops_size * 2);
        kd.internalBuffers.push_back(4 * group_num * ops_size * 2);
        kd.internalBuffers.push_back(ops_size * elem_size);
        kd.internalBufferDataType = prim_params.inputs[0].GetDType();
    };
}

KernelsData ArgMaxMinKernelAxis::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }
    const arg_max_min_params& orgParams = static_cast<const arg_max_min_params&>(params);
    bool is_dynamic = orgParams.has_dynamic_tensors();

    auto dispatchData = SetDefault(orgParams);
    KernelData kd = KernelData::Default<arg_max_min_params>(params);
    GetUpdateDispatchDataFunc(kd);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

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
                     orgParams.outputs_num,
                     orgParams.is_shape_agnostic);

    if (is_dynamic) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        kd.internalBuffers.push_back(orgParams.inputs[0].PhysicalSizeInBytes());
        kd.internalBuffers.push_back(orgParams.inputs[0].PhysicalSizeInBytes());
        kd.internalBuffers.push_back(orgParams.inputs[0].PhysicalSizeInBytes());
        kd.internalBufferDataType = orgParams.inputs[0].GetDType();
    }

    return {kd};
}

KernelsPriority ArgMaxMinKernelAxis::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}

JitConstants ArgMaxMinKernelAxis::GetJitConstants(const arg_max_min_params& params) const {
    auto jit = ArgMaxMinKernelBase::GetJitConstants(params);

    if (params.has_dynamic_tensors()) {
        const std::string operation_num = getOperationNumberString(params);
        jit.AddConstant(MakeJitConstant("OPERATION_NUM", operation_num));
    } else {
        const size_t operation_num = getOperationNumber(params);
        jit.AddConstant(MakeJitConstant("OPERATION_NUM", operation_num));
    }
    if (params.argMaxMinSortType == ArgMaxMinSortType::VALUE)
        jit.AddConstant(MakeJitConstant("SORT_BY_VALUE", 1));
    else
        jit.AddConstant(MakeJitConstant("SORT_BY_INDEX", 1));

    if (params.values_first)
        jit.AddConstant(MakeJitConstant("TOP_K_ORDER", 1));

    return jit;
}
}  // namespace kernel_selector
