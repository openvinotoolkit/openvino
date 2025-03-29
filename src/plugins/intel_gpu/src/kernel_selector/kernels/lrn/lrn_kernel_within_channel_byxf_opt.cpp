// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lrn_kernel_within_channel_byxf_opt.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
ParamsKey LRNKernelWithinChannelByxfOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableLRNMode(LRNMode::WITHIN_CHANNEL);
    k.EnableLRNKernelDividerMode(KernelDividerMode::DYNAMIC);
    k.EnableLRNKernelDividerMode(KernelDividerMode::FIXED);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

JitConstants LRNKernelWithinChannelByxfOpt::GetJitConstants(const lrn_params& params,
                                                            const LRNKernelBase::DispatchData& dispatchData) const {
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
        FusedOpsConfiguration conf = {"", {"b", "f + i", "y", "x"}, "lrn_result", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

LRNKernelWithinChannelByxfOpt::Parent::DispatchData LRNKernelWithinChannelByxfOpt::SetDefault(
    const lrn_params& params) const {
    DispatchData dispatchData = Parent::SetDefault(params);
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X, Tensor::DataChannelName::Y },
                                                                     { Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::BATCH }};

    const auto& out = params.outputs[0];

    dispatchData.gws = { out.X().v * out.Y().v, CeilDiv(out.Feature().v, 8), out.Batch().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

bool LRNKernelWithinChannelByxfOpt::Validate(const Params& p) const {
    if (!LRNKernelBase::Validate(p)) {
        return false;
    }
    const lrn_params& params = static_cast<const lrn_params&>(p);
    if (params.inputs[0].Feature().v % 8 != 0) {
        return false;
    }
    return true;
}

KernelsData LRNKernelWithinChannelByxfOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority LRNKernelWithinChannelByxfOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_7;
}
}  // namespace kernel_selector
