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
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const arg_max_min_params& params = static_cast<const arg_max_min_params&>(p);

    if (params.inputs.size() > 1) {
        if (params.inputs[1].PitchesDifferFromLogicalDims() || params.outputs[0].PitchesDifferFromLogicalDims())
            DO_NOT_USE_THIS_KERNEL(p.layerID);
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
        dispatchData.lws = { 1, 1, 1};
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

        auto bufferSizeVec = GetTempBufferVec(prim_params, 1);
        const size_t total_buffer_size = std::accumulate(bufferSizeVec.begin(), bufferSizeVec.end(), 0);
        if (total_buffer_size < params.engineInfo.maxLocalMemSize) {
            kd.kernels[0].params.local_memory_args.clear();
            kd.kernels[0].params.local_memory_args.push_back(bufferSizeVec.at(0));
            kd.kernels[0].params.local_memory_args.push_back(bufferSizeVec.at(1));
            kd.kernels[0].params.local_memory_args.push_back(bufferSizeVec.at(2));
        } else {
            kd.internalBuffers.clear();
            kd.internalBuffers.push_back(bufferSizeVec.at(0));
            kd.internalBuffers.push_back(bufferSizeVec.at(1));
            kd.internalBuffers.push_back(bufferSizeVec.at(2));
            kd.internalBufferDataType = prim_params.inputs[0].GetDType();
        }
    };
}


std::vector<size_t> ArgMaxMinKernelAxis::GetTempBufferVec(const arg_max_min_params& params, bool dynamic) const {
    std::vector<size_t> buffer_size;

    const size_t elem_size = params.inputs[0].ElementSize();
    const size_t iav_type_size = elem_size + 4;
    const size_t sort_size = getSortSize(params);
    const size_t ops_size = getOperationNumber(params);
    const size_t group_size = params.topK >= 8 ? params.topK : 8;
    const size_t group_num = ((sort_size - 1) / group_size) + 1;

    if (dynamic) {
        const size_t buffer0_size = iav_type_size * sort_size * 2;
        const size_t buffer1_size = iav_type_size * sort_size * 2;
        const size_t buffer2_size = group_num;

        buffer_size.assign({buffer0_size, buffer1_size, buffer2_size});
    } else {
        const size_t buffer0_size = iav_type_size * sort_size * ops_size * 2;
        const size_t buffer1_size = 4 * group_num * ops_size * 2;
        const size_t buffer2_size = ops_size * elem_size;

        buffer_size.assign({buffer0_size, buffer1_size, buffer2_size});
    }

    return buffer_size;
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
    auto bufferSizeVec = GetTempBufferVec(orgParams, orgParams.has_dynamic_tensors());

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

    const size_t total_buffer_size = std::accumulate(bufferSizeVec.begin(),
            bufferSizeVec.end(), 0);

    if (is_dynamic) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 2});
        kd.kernels[0].params.local_memory_args.push_back(bufferSizeVec.at(0));
        kd.kernels[0].params.local_memory_args.push_back(bufferSizeVec.at(1));
        kd.kernels[0].params.local_memory_args.push_back(bufferSizeVec.at(2));
    } else if (total_buffer_size > orgParams.engineInfo.maxLocalMemSize) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        kd.internalBuffers.push_back(bufferSizeVec.at(0));
        kd.internalBuffers.push_back(bufferSizeVec.at(1));
        kd.internalBuffers.push_back(bufferSizeVec.at(2));
        kd.internalBufferDataType = orgParams.inputs[0].GetDType();
    }

    return {kd};
}

KernelsPriority ArgMaxMinKernelAxis::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}

JitConstants ArgMaxMinKernelAxis::GetJitConstants(const arg_max_min_params& params) const {
    auto jit = ArgMaxMinKernelBase::GetJitConstants(params);
    auto bufferSizeVec = GetTempBufferVec(params, params.has_dynamic_tensors());

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

    const size_t total_buffer_size = std::accumulate(bufferSizeVec.begin(),
            bufferSizeVec.end(), 0);

    // For dynamic case, total_buffer_size will be 0.  If larger buffer are
    // requested during runtime and greater than local memory size
    // the kernel will have incorrect data
    // total_buffer_size only for static case
    if (params.has_dynamic_tensors()) {
        jit.AddConstant(MakeJitConstant("USE_LOCAL_MEMORY", 1));
    } else if (total_buffer_size > params.engineInfo.maxLocalMemSize) {
        jit.AddConstant(MakeJitConstant("USE_INTERNAL_BUFFERS", 1));
    }

    return jit;
}
}  // namespace kernel_selector
