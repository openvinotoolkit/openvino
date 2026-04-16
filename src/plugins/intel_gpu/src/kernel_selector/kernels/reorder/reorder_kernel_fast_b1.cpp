// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel_fast_b1.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderKernelFastBatch1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::fyxb);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableInputLayout(DataLayout::bs_f_bsv8__af8);
    k.EnableInputLayout(DataLayout::bs_f_bsv16__af8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);

    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::fyxb);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bs_f_bsv8__af8);
    k.EnableOutputLayout(DataLayout::bs_f_bsv16__af8);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

bool ReorderKernelFastBatch1::Validate(const Params& p) const {
    if (!ReorderKernelBase::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const reorder_params& params = static_cast<const reorder_params&>(p);

    if (params.outputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.inputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    return true;
}

// Returns the feature sub-vector size for blocked output formats, or 0 for non-blocked formats.
static size_t GetOutputFeatureBlockSize(DataLayout layout) {
    switch (layout) {
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::b_fs_zyx_fsv16:
        return 16;
    case DataLayout::b_fs_yx_fsv32:
    case DataLayout::b_fs_zyx_fsv32:
        return 32;
    default:
        return 0;
    }
}

JitConstants ReorderKernelFastBatch1::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));

    KernelData kd = KernelData::Default<reorder_params>(params);
    reorder_params& newParams = *static_cast<reorder_params*>(kd.params.get());

    const auto& input = newParams.inputs[0];
    const auto& output = newParams.outputs[0];

    // For blocked output formats (e.g., b_fs_yx_fsv16), when the feature count is not
    // aligned to the block size, we must zero-fill the padding positions. Otherwise,
    // uninitialized memory (potentially containing NaN) in padding positions can corrupt
    // subsequent computations (since NaN * 0 = NaN in IEEE 754).
    size_t fsv = GetOutputFeatureBlockSize(output.GetLayout());
    size_t feature_count = output.Feature().v;
    if (fsv > 0 && (feature_count % fsv) != 0) {
        size_t padded_features = Align(feature_count, fsv);
        size_t spatial_size = output.LogicalSize() / (output.Batch().v * feature_count);
        size_t padded_count = output.Batch().v * padded_features * spatial_size;
        jit.AddConstant(MakeJitConstant("ELEMENTS_COUNT", padded_count));
        jit.AddConstant(MakeJitConstant("PADDED_FEATURE_NUM", padded_features));
        jit.AddConstant(MakeJitConstant("FILL_FEATURE_PADDING", 1));
    } else {
        jit.AddConstant(MakeJitConstant("ELEMENTS_COUNT", input.LogicalSize()));
    }

    if (input.GetLayout() == output.GetLayout() && input.SameDimsSizes(output) &&
        !input.PitchesDifferFromLogicalDims() && !output.PitchesDifferFromLogicalDims() &&
        input.GetDType() != output.GetDType() && !params.has_padded_output &&
        params.mode == MeanSubtractMode::NONE) {
        jit.AddConstant(MakeJitConstant("CHANGE_DATA_TYPE_ONLY", 1));
    }

    return jit;
}

ReorderKernelFastBatch1::DispatchData ReorderKernelFastBatch1::SetDefault(const reorder_params& params) const {
    DispatchData dispatchData;

    const auto& output = params.outputs[0];

    unsigned int gws = (unsigned int)output.LogicalSize();

    // For blocked output formats with unaligned features, expand GWS to cover padding positions
    size_t fsv = GetOutputFeatureBlockSize(output.GetLayout());
    size_t feature_count = output.Feature().v;
    if (fsv > 0 && (feature_count % fsv) != 0) {
        size_t padded_features = Align(feature_count, fsv);
        size_t spatial_size = output.LogicalSize() / (output.Batch().v * feature_count);
        gws = (unsigned int)(output.Batch().v * padded_features * spatial_size);
    }

    dispatchData.gws[0] = Align(gws, 32);
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = 32;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsData ReorderKernelFastBatch1::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::REORDER);

    const reorder_params& orgParams = static_cast<const reorder_params&>(params);

    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderKernelFastBatch1::GetKernelsPriority(const Params& params) const {
    const reorder_params& orgParams = static_cast<const reorder_params&>(params);
    const auto& input = orgParams.inputs[0];
    const auto& output = orgParams.outputs[0];

    return input.Batch().v == 1 && output.Batch().v == 1 ? FORCE_PRIORITY_6 : DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
