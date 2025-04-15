// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_weights_image_fyx_b_kernel.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
ParamsKey ReorderWeightsImage_fyx_b_Kernel::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputWeightsType(WeightsType::F16);
    k.EnableOutputWeightsType(WeightsType::F32);
    k.EnableInputWeightsLayout(WeightsLayout::oiyx);
    k.EnableOutputWeightsLayout(WeightsLayout::image_2d_weights_c4_fyx_b);
    k.EnableWinogradReorder();
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

ReorderWeightsImage_fyx_b_Kernel::DispatchData ReorderWeightsImage_fyx_b_Kernel::SetDefault(const reorder_weights_params& params) const {
    const auto& out = params.output;

    DispatchData dispatchData;

    dispatchData.gws = { out.OFM().v, Align(out.X().v * out.Y().v * out.IFM().v, 4) / 4, 1 };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData ReorderWeightsImage_fyx_b_Kernel::GetKernelsData(const Params& params) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderWeightsImage_fyx_b_Kernel::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_4;
}
}  // namespace kernel_selector
