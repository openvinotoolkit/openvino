// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_weights_bf16.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderWeightsBF16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::BF16);
    k.EnableOutputWeightsType(WeightsType::INT8);
    k.EnableOutputWeightsType(WeightsType::F16);
    k.EnableOutputWeightsType(WeightsType::F32);
    k.EnableOutputWeightsType(WeightsType::INT32);
    k.EnableAllInputWeightsLayout();
    k.EnableAllOutputWeightsLayout();
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableRotateReorder();
    return k;
}

KernelsData ReorderWeightsBF16::GetKernelsData(const Params& params) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderWeightsBF16::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}
}  // namespace kernel_selector
