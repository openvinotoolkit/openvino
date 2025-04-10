// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_weights_image_winograd_6x3_kernel.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderWeightsImageWinograd6x3Kernel::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputWeightsType(WeightsType::F16);
    k.EnableOutputWeightsType(WeightsType::F32);
    k.EnableAllInputWeightsLayout();
    k.EnableOutputWeightsLayout(WeightsLayout::image_2d_weights_winograd_6x3_s1_fbxyb);
    k.EnableOutputWeightsLayout(WeightsLayout::image_2d_weights_winograd_6x3_s1_xfbyb);
    k.EnableWinogradReorder();
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

ReorderWeightsImageWinograd6x3Kernel::DispatchData ReorderWeightsImageWinograd6x3Kernel::SetDefault(
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

KernelsData ReorderWeightsImageWinograd6x3Kernel::GetKernelsData(const Params& params) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderWeightsImageWinograd6x3Kernel::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_4;
}
}  // namespace kernel_selector
