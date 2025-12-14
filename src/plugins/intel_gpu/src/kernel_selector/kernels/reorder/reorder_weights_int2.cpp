// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_weights_int2.h"
#include "kernel_selector_common.h"
#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"
#include "common_types.h"

namespace kernel_selector {

ParamsKey ReorderWeightsKernelInt2::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::INT2);
    k.EnableInputWeightsType(WeightsType::UINT2);
    k.EnableOutputWeightsType(WeightsType::UINT2);
    k.EnableOutputWeightsType(WeightsType::INT2);
    k.EnableInputWeightsLayout(WeightsLayout::oiyx);
    k.EnableInputWeightsLayout(WeightsLayout::ioyx);
    k.EnableOutputWeightsLayout(WeightsLayout::oiyx);
    k.EnableOutputWeightsLayout(WeightsLayout::ioyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

KernelsData ReorderWeightsKernelInt2::GetKernelsData(const Params& params) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams);
}

ReorderWeightsKernelInt2::DispatchData ReorderWeightsKernelInt2::SetDefault(const reorder_weights_params& params) const {
    DispatchData dispatchData;

    const auto& output = params.output;

    // Divide one of the dimensions by 2 to save with byte granularity
    if (output.GetLayout() == WeightsLayout::oiyx || output.GetLayout() == WeightsLayout::ioyx) {
        auto dims = output.GetDims();
        bool has_pads = std::any_of(dims.begin(), dims.end(), [](const kernel_selector::Tensor::Dim& d) {
            return d.pad.Total() != 0;
        });
        if (has_pads) {
            dispatchData.gws = { CeilDiv(output.PhysicalSize(), 4), 1, 1 };
        } else {
            dispatchData.gws = { CeilDiv(output.LogicalSize(), 4), 1, 1 };
        }
    }
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

JitConstants ReorderWeightsKernelInt2::GetJitConstants(const reorder_weights_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);


    return jit;
}

bool ReorderWeightsKernelInt2::Validate(const Params& params) const {
    const auto& p = static_cast<const reorder_weights_params&>(params);
    const auto& input = p.input;
    const auto& output = p.output;
    if((input.GetLayout() == WeightsLayout::oiyx && output.GetLayout() == WeightsLayout::oiyx) ||
        (input.GetLayout() == WeightsLayout::ioyx && output.GetLayout() == WeightsLayout::ioyx)) {
        return true;
    }
    return false;
}

KernelsPriority ReorderWeightsKernelInt2::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
