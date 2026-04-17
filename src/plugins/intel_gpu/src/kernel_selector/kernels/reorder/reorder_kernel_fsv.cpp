// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel_fsv.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {

static size_t GetFsv(DataLayout layout) {
    switch (layout) {
    case DataLayout::b_fs_yx_fsv4:
    case DataLayout::b_fs_zyx_fsv4:
        return 4;
    case DataLayout::b_fs_yx_fsv8:
    case DataLayout::b_fs_zyx_fsv8:
        return 8;
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

static bool IsSingleBlockedFsv(DataLayout layout) {
    return GetFsv(layout) != 0;
}

ParamsKey ReorderKernel_fsv::GetSupportedKey() const {
    ParamsKey k;

    k.EnableAllInputDataType();
    k.EnableAllOutputDataType();

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv4);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv8);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);

    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv8);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv8);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);

    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDynamicShapesSupport();

    return k;
}

bool ReorderKernel_fsv::Validate(const Params& p) const {
    if (!ReorderKernelBase::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const reorder_params& params = static_cast<const reorder_params&>(p);
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    if (!IsSingleBlockedFsv(input.GetLayout()) || !IsSingleBlockedFsv(output.GetLayout())) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    if (input.GetDims().size() != output.GetDims().size()) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    size_t in_fsv = GetFsv(input.GetLayout());
    size_t out_fsv = GetFsv(output.GetLayout());
    if (in_fsv == out_fsv) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    // Padding not supported
    if (input.X().pad.before != 0 || input.X().pad.after != 0 ||
        input.Y().pad.before != 0 || input.Y().pad.after != 0 ||
        input.Z().pad.before != 0 || input.Z().pad.after != 0 ||
        input.Feature().pad.before != 0 || input.Feature().pad.after != 0 ||
        input.Batch().pad.before != 0 || input.Batch().pad.after != 0) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    if (output.X().pad.before != 0 || output.X().pad.after != 0 ||
        output.Y().pad.before != 0 || output.Y().pad.after != 0 ||
        output.Z().pad.before != 0 || output.Z().pad.after != 0 ||
        output.Feature().pad.before != 0 || output.Feature().pad.after != 0 ||
        output.Batch().pad.before != 0 || output.Batch().pad.after != 0) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    return true;
}

CommonDispatchData ReorderKernel_fsv::SetDefault(const reorder_params& params) const {
    CommonDispatchData dispatchData;

    const auto& input = params.inputs[0];
    const size_t out_fsv = GetFsv(params.outputs[0].GetLayout());

    // GWS[0] = x
    // GWS[1] = y * z
    // GWS[2] = b * fs_out (feature slices in output blocking)
    size_t x = input.X().v;
    size_t y = input.Y().v;
    size_t z = input.Z().v;
    size_t b = input.Batch().v;
    size_t f = input.Feature().v;

    dispatchData.gws = { x, y * z, b * CeilDiv(f, out_fsv) };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

JitConstants ReorderKernel_fsv::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);

    const size_t in_fsv = GetFsv(params.inputs[0].GetLayout());
    const size_t out_fsv = GetFsv(params.outputs[0].GetLayout());
    const size_t ndims = params.inputs[0].GetDims().size();

    jit.AddConstant(MakeJitConstant("IN_FSV", in_fsv));
    jit.AddConstant(MakeJitConstant("OUT_FSV", out_fsv));

    if (ndims == 5) {
        jit.AddConstant(MakeJitConstant("INPUT0_DIMS_5", 1));
    }

    // Vectorized path: when both fsv sizes are >= 16 and one is a multiple of the other
    const size_t min_fsv = std::min(in_fsv, out_fsv);
    const size_t max_fsv = std::max(in_fsv, out_fsv);
    if (min_fsv >= 16 && max_fsv % min_fsv == 0) {
        jit.AddConstant(MakeJitConstant("FSV_VECTORIZED", 1));
        jit.AddConstant(MakeJitConstant("VEC_SIZE", min_fsv));
        jit.AddConstant(MakeJitConstant("RATIO", max_fsv / min_fsv));
    }

    if (params.is_shape_agnostic) {
        jit.AddConstant(MakeJitConstant("IN_FEATURE_SLICE_NUM",
            "((INPUT0_FEATURE_NUM + " + std::to_string(in_fsv) + " - 1) / " + std::to_string(in_fsv) + ")"));
        jit.AddConstant(MakeJitConstant("OUT_FEATURE_SLICE_NUM",
            "((INPUT0_FEATURE_NUM + " + std::to_string(out_fsv) + " - 1) / " + std::to_string(out_fsv) + ")"));
    } else {
        const size_t f = params.inputs[0].Feature().v;
        jit.AddConstant(MakeJitConstant("IN_FEATURE_SLICE_NUM", CeilDiv(f, in_fsv)));
        jit.AddConstant(MakeJitConstant("OUT_FEATURE_SLICE_NUM", CeilDiv(f, out_fsv)));
    }

    return jit;
}

KernelsData ReorderKernel_fsv::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::REORDER);

    const reorder_params& orgParams = static_cast<const reorder_params&>(params);

    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderKernel_fsv::GetKernelsPriority(const Params& p) const {
    return FORCE_PRIORITY_4;
}

}  // namespace kernel_selector
