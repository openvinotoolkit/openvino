// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "sparse_fill_empty_rows_kernel_ref.h"
#include <kernel_selector_utils.h>

namespace kernel_selector {

ParamsKey SparseFillEmptyRowsKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

namespace {
SparseFillEmptyRowsKernelRef::DispatchData SetDefault(const sparse_fill_empty_rows_params& params) {
    SparseFillEmptyRowsKernelRef::DispatchData dispatchData;
    const auto rows_count = params.outputs[2].GetDims()[3].v;  // empty_row_indicator has size [rows]
    dispatchData.gws[0] = rows_count;
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}
} // anonymous namespace

void SparseFillEmptyRowsKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const sparse_fill_empty_rows_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = SkipKernelExecution(prim_params, 0);
    };
}

KernelsData SparseFillEmptyRowsKernelRef::GetKernelsData(const Params &params) const {
    if (!Validate(params)) {
        return {};
    }
    KernelData kernel_data = KernelData::Default<sparse_fill_empty_rows_params>(params);
    kernel_data.kernels[0].skip_execution = SkipKernelExecution(static_cast<const sparse_fill_empty_rows_params&>(params));
    sparse_fill_empty_rows_params &new_params = dynamic_cast<sparse_fill_empty_rows_params&>(*kernel_data.params.get());
    auto sparse_fill_empty_rows_specific_jit = GetJitConstants(new_params);
    auto dispatch_data = SetDefault(new_params);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params);
    auto jit = CreateJit(kernelName, sparse_fill_empty_rows_specific_jit, entry_point);
    FillCLKernelData(kernel_data.kernels[0],
        dispatch_data,
        params.engineInfo,
        kernelName,
        jit,
        entry_point,
        EXE_MODE_DEFAULT,               // exeMode
        false,                          // weights
        false,                          // bias
        4,                              // number_of_inputs
        0,                              // number_of_inputs_for_fused_prims
        3,                              // number_of_outputs
        params.is_shape_agnostic);      // is_dynamic
    GetUpdateDispatchDataFunc(kernel_data);
    return {kernel_data};
}

float SparseFillEmptyRowsKernelRef::GetKernelsPriority(const Params &params) const {
    return FORCE_PRIORITY_1;
}

bool SparseFillEmptyRowsKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SPARSE_FILL_EMPTY_ROWS) {
        return false;
    }
    return true;
}

JitConstants SparseFillEmptyRowsKernelRef::GetJitConstants(const sparse_fill_empty_rows_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    if (!params.is_shape_agnostic) {
        jit.AddConstant(MakeJitConstant("NUM_INDICES", params.inputs[2].LogicalSize() / 2));
    }
    return jit;
}

bool SparseFillEmptyRowsKernelRef::SkipKernelExecution(const sparse_fill_empty_rows_params& params, size_t kernel_id) const {
    // If indices tensor size has changed the kernel has to be executed
    if (params.inputs[2].LogicalSize() != params.outputs[0].LogicalSize()) {
        return false;
    }
    return KernelData::SkipKernelExecution(params);
}
}  // namespace kernel_selector
