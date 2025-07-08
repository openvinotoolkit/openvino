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
    //k.EnableDynamicShapesSupport();
    k.EnableDifferentTypes();
    return k;
}

namespace {
SparseFillEmptyRowsKernelRef::DispatchData SetDefault(const sparse_fill_empty_rows_params& params) {
    SparseFillEmptyRowsKernelRef::DispatchData dispatchData;
    //dispatchData.gws[0] = params.outputs[0].LogicalSize();
    dispatchData.gws[0] = 1;
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;
    //dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    dispatchData.lws = {1, 1, 1};

    return dispatchData;
}
} // anonymous namespace

KernelsData SparseFillEmptyRowsKernelRef::GetKernelsData(const Params &params) const {
    if (!Validate(params)) {
        return {};
    }
    KernelData kernel_data = KernelData::Default<sparse_fill_empty_rows_params>(params);
    sparse_fill_empty_rows_params &new_params = dynamic_cast<sparse_fill_empty_rows_params&>(*kernel_data.params.get());
    auto dispatch_data = SetDefault(new_params);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params);
    auto sparse_fill_empty_rows_specific_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, sparse_fill_empty_rows_specific_jit, entry_point);
    FillCLKernelData(kernel_data.kernels[0],
        dispatch_data,
        params.engineInfo,
        kernelName,
        jit,
        entry_point,
        EXE_MODE_DEFAULT,  // exeMode
        false,             // weights
        false,             // bias
        4,                 // number_of_inputs
        0,                 // number_of_inputs_for_fused_prims
        3);                // number_of_outputs
    return {kernel_data};
}

float SparseFillEmptyRowsKernelRef::GetKernelsPriority(const Params &params) const {
    return FORCE_PRIORITY_1;
}

bool SparseFillEmptyRowsKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SPARSE_FILL_EMPTY_ROWS) {
        return false;
    }

    const sparse_fill_empty_rows_params &params = static_cast<const sparse_fill_empty_rows_params&>(p);
    if (params.inputs.size() != 4)
        return false;

    if (params.outputs.size() != 3)
        return false;

    return true;
}

JitConstants SparseFillEmptyRowsKernelRef::GetJitConstants(const sparse_fill_empty_rows_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    return jit;
}

}  // namespace kernel_selector
