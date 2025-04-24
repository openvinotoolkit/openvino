// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_weights_winograd_2x3_kernel.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderWeightsWinograd2x3Kernel::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputWeightsType(WeightsType::F16);
    k.EnableOutputWeightsType(WeightsType::F32);
    k.EnableAllInputWeightsLayout();
    k.EnableOutputWeightsLayout(WeightsLayout::winograd_2x3_s1_weights);
    k.EnableOutputWeightsLayout(WeightsLayout::winograd_2x3_s1_fused_weights);
    k.EnableWinogradReorder();
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

ReorderWeightsWinograd2x3Kernel::DispatchData ReorderWeightsWinograd2x3Kernel::SetDefault(
    const reorder_weights_params& params) const {
    DispatchData dispatchData;

    const auto& input = params.input;

    dispatchData.gws[0] = 1;
    dispatchData.gws[1] = 3;
    dispatchData.gws[2] = static_cast<size_t>(input.IFM().v * input.OFM().v);

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 32;

    return dispatchData;
}

KernelsData ReorderWeightsWinograd2x3Kernel::GetKernelsData(const Params& params) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderWeightsWinograd2x3Kernel::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_4;
}
}  // namespace kernel_selector
