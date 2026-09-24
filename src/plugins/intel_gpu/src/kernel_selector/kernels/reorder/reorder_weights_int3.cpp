// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_weights_int3.h"
#include "kernel_selector_common.h"
#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"
#include "common_types.h"

namespace kernel_selector {

static constexpr size_t osv = 16;
static constexpr size_t isv = 32;

ParamsKey ReorderWeightsKernelInt3::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::UINT3);
    k.EnableOutputWeightsType(WeightsType::UINT3);
    k.EnableInputWeightsLayout(WeightsLayout::oiyx);
    k.EnableOutputWeightsLayout(WeightsLayout::os_is_yx_osv16_isv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

KernelsData ReorderWeightsKernelInt3::GetKernelsData(const Params& params) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams);
}

ReorderWeightsKernelInt3::DispatchData ReorderWeightsKernelInt3::SetDefault(const reorder_weights_params& params) const {
    DispatchData dispatchData;

    const auto& output = params.output;

    // One work item repacks one granule: a single output channel by isv input channels.
    dispatchData.gws = { Align(output.OFM().v, osv), CeilDiv(output.IFM().v, isv), 1 };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

JitConstants ReorderWeightsKernelInt3::GetJitConstants(const reorder_weights_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    jit.AddConstant(MakeJitConstant("CHUNKS_K", CeilDiv(params.output.IFM().v, isv)));
    return jit;
}

bool ReorderWeightsKernelInt3::Validate(const Params& params) const {
    const auto& p = static_cast<const reorder_weights_params&>(params);
    const auto& input = p.input;
    const auto& output = p.output;

    if (input.LogicalSize() != input.OFM().v * input.IFM().v ||
        output.LogicalSize() != output.OFM().v * output.IFM().v)
        return false;

    return input.GetLayout() == WeightsLayout::oiyx &&
           output.GetLayout() == WeightsLayout::os_is_yx_osv16_isv32;
}

KernelsPriority ReorderWeightsKernelInt3::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
