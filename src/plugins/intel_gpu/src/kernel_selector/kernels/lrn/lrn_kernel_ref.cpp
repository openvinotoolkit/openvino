// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lrn_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
ParamsKey LRNKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableLRNMode(LRNMode::WITHIN_CHANNEL);
    k.EnableLRNMode(LRNMode::ACROSS_CHANNEL);
    k.EnableLRNKernelDividerMode(KernelDividerMode::DYNAMIC);
    k.EnableLRNKernelDividerMode(KernelDividerMode::FIXED);
    k.EnableDifferentTypes();
    return k;
}

JitConstants LRNKernelRef::GetJitConstants(const lrn_params& params, const LRNKernelRef::Parent::DispatchData& dispatchData) const {
    const uint32_t round_norm_size = (params.localSize / 2) * 2 + 1;
    uint32_t numElement = round_norm_size * round_norm_size;
    const auto& input_dt = params.inputs[0].GetDType();

    if (params.normMode == LRNMode::ACROSS_CHANNEL) {
        numElement = round_norm_size;
    }

    const float num_element_div = 1.f / static_cast<float>(numElement);

    JitConstants jit = Parent::GetJitConstants(params, dispatchData);
    jit.AddConstants({
        MakeJitConstant("NUM_ELEMENTS_DIV", num_element_div),
        MakeJitConstant("GWS_BATCH", 2),
        MakeJitConstant("GWS_FEATURE", 1),
        MakeJitConstant("GWS_YX", 0),
    });

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {"", {"b", "f", "y", "x"}, "lrn_result", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

LRNKernelRef::Parent::DispatchData LRNKernelRef::SetDefault(const lrn_params& params) const {
    DispatchData dispatchData = Parent::SetDefault(params);
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X, Tensor::DataChannelName::Y },
                                                                     { Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::BATCH }};

    const auto& out = params.outputs[0];

    dispatchData.gws = { out.X().v * out.Y().v, out.Feature().v, out.Batch().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

KernelsData LRNKernelRef::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority LRNKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
