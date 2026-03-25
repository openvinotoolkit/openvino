// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_b_f_axes.h"

#include <string>

#include "common_tools.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

// Vector width chosen to make each vload/vstore a 16-byte transaction.
static size_t GetVecWidth(const permute_params& params) {
    switch (params.inputs[0].GetDType()) {
        case Datatype::F16:
        case Datatype::INT16:
        case Datatype::UINT16:
            return 8;
        case Datatype::F32:
        case Datatype::INT32:
            return 4;
        case Datatype::INT8:
        case Datatype::UINT8:
            return 16;
        case Datatype::INT64:
            return 2;
        default:
            return 4;
    }
}

ParamsKey PermuteKernel_b_f_axes::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableDifferentTypes();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants PermuteKernel_b_f_axes::GetJitConstants(const permute_params& params,
                                                     const CommonDispatchData& /*dispatchData*/) const {
    auto jit = Parent::GetJitConstants(params, {});

    const size_t vec_width = GetVecWidth(params);
    const size_t x_size    = params.inputs[0].X().v;
    const size_t x_tiles   = x_size / vec_width;
    const size_t x_rem     = x_size % vec_width;

    jit.AddConstant(MakeJitConstant("VEC_WIDTH",       vec_width));
    jit.AddConstant(MakeJitConstant("X_TILES",         x_tiles));
    jit.AddConstant(MakeJitConstant("X_REMAINDER_SIZE", x_rem));

    jit.AddConstant(MakeJitConstant("INPUTVTYPE",  "CAT(INPUT0_TYPE, VEC_WIDTH)"));
    jit.AddConstant(MakeJitConstant("OUTPUTVTYPE", "CAT(OUTPUT_TYPE, VEC_WIDTH)"));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> output_order;
        switch (params.inputs[0].GetDims().size()) {
            case 4: output_order = {"b", "f", "y", "x"}; break;
            case 5: output_order = {"b", "f", "z", "y", "x"}; break;
            case 6: output_order = {"b", "f", "w", "z", "y", "x"}; break;
            default: break;
        }
        FusedOpsConfiguration conf = {"", output_order, "input_var", params.inputs[0].GetDType(), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

CommonDispatchData PermuteKernel_b_f_axes::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatchData;

    const auto& in        = params.inputs[0];
    const auto in_layout  = in.GetLayout();
    const auto out_layout = params.outputs[0].GetLayout();
    const size_t vec_width = GetVecWidth(params);
    const size_t x_tiles = CeilDiv(in.X().v, vec_width);

    size_t spatial_outer = 1;
    if (in.GetDims().size() >= 5) spatial_outer *= in.Z().v;
    if (in.GetDims().size() >= 6) spatial_outer *= in.W().v;

    // F is looped inside the kernel; GWS[2] covers B only.
    dispatchData.gws = {x_tiles, in.Y().v * spatial_outer, in.Batch().v};

    const std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {
        {Tensor::DataChannelName::X},
        {Tensor::DataChannelName::Y, Tensor::DataChannelName::Z, Tensor::DataChannelName::W},
        {Tensor::DataChannelName::BATCH}};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo,
                                                     in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

bool PermuteKernel_b_f_axes::Validate(const Params& p) const {
    if (!Parent::Validate(p)) DO_NOT_USE_THIS_KERNEL(p.layerID);

    const permute_params& params = static_cast<const permute_params&>(p);

    if (params.outputs[0].PitchesDifferFromLogicalDims() || params.inputs[0].PitchesDifferFromLogicalDims())
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (!SimpleLayout(params.inputs[0].GetLayout()))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.inputs[0].GetLayout() != params.outputs[0].GetLayout())
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Only accept [1, 0, 2, ...] — B <-> F swap with spatial axes unchanged.
    const auto& order = params.order;
    const size_t ndim = order.size();
    if (ndim < 3 || ndim > 6)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (order[0] != 1 || order[1] != 0)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    for (size_t i = 2; i < ndim; ++i) {
        if (order[i] != static_cast<uint16_t>(i))
            DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    return true;
}

KernelsPriority PermuteKernel_b_f_axes::GetKernelsPriority(const Params& params) const {
    KernelData kd = KernelData::Default<permute_params>(params);
    permute_params& newParams = *static_cast<permute_params*>(kd.params.get());

    const size_t vec_width = GetVecWidth(newParams);
    const size_t x_size    = newParams.inputs[0].X().v;

    if (x_size >= vec_width * 2 && (x_size % vec_width == 0))
        return FORCE_PRIORITY_2;
    if (x_size >= vec_width)
        return FORCE_PRIORITY_3;

    return FORCE_PRIORITY_4;
}

}  // namespace kernel_selector
