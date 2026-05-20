// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_kernel_bfyx_cubic_opt.h"
#include <vector>
#include <kernel_selector_utils.h>

namespace kernel_selector {

size_t ResampleKernelBfyxCubicOpt::GetOptimalBlockSize(const resample_params& params) const {
    std::vector<size_t> block_width = {16, 8, 4, 2, 1};
    for (auto& w : block_width) {
        if (params.outputs[0].X().v % w == 0) {
            return w;
        }
    }
    return 1;
}

ParamsKey ResampleKernelBfyxCubicOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableResampleType(ResampleType::CUBIC);
    return k;
}

ResampleKernelBase::DispatchData ResampleKernelBfyxCubicOpt::SetDefault(const resample_params& arg) const {
    DispatchData dispatchData;
    auto in_layout = arg.inputs[0].GetLayout();
    auto out_layout = arg.outputs[0].GetLayout();

    const auto& out = arg.outputs[0];
    auto opt_x_block_size = GetOptimalBlockSize(arg);

    dispatchData.gws[0] = CeilDiv(out.X().v, opt_x_block_size) * out.Y().v;
    dispatchData.gws[1] = out.Feature().v;
    dispatchData.gws[2] = out.Batch().v;

    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {
        {Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
        {Tensor::DataChannelName::FEATURE},
        {Tensor::DataChannelName::BATCH}};

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

KernelsPriority ResampleKernelBfyxCubicOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}

bool ResampleKernelBfyxCubicOpt::Validate(const Params& p) const {
    if (!Parent::Validate(p))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const resample_params& params = static_cast<const resample_params&>(p);

    // Only 4D tensors (no Z axis).
    if (params.inputs[0].Dimentions() != 4)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Only spatial axes (Y, X) may be resized.
    for (const auto& axis : params.axes) {
        if (axis != InterpolateAxis::Y && axis != InterpolateAxis::X)
            DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    // Non-spatial dimensions (batch and feature) must not be resized.
    const auto& in = params.inputs[0];
    const auto& out = params.outputs[0];
    if (in.Batch().v != out.Batch().v || in.Feature().v != out.Feature().v)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    return true;
}

JitConstants ResampleKernelBfyxCubicOpt::GetJitConstants(const resample_params& params) const {
    auto jit = Parent::GetJitConstants(params);

    auto opt_x_block_size = GetOptimalBlockSize(params);
    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", opt_x_block_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.outputs[0].X().v, opt_x_block_size)));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order = {"batch", "OF_ID", "oy", "ox"};
        FusedOpsConfiguration conf = {"", idx_order, "interp_val", GetAccumulatorType(params), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

KernelsData ResampleKernelBfyxCubicOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

}  // namespace kernel_selector
