// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_weights_binary_kernel.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
ParamsKey ReorderWeightsBinaryKernel::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::BINARY);
    k.EnableOutputWeightsType(WeightsType::BINARY);
    k.EnableInputWeightsLayout(WeightsLayout::oiyx);
    k.EnableOutputWeightsLayout(WeightsLayout::os_is_yx_osv32_isv32p);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

ReorderWeightsBinaryKernel::DispatchData ReorderWeightsBinaryKernel::SetDefault(
    const reorder_weights_params& params) const {
    const auto& out = params.output;

    DispatchData dispatchData;

    dispatchData.gws = { out.OFM().v, CeilDiv(out.IFM().v, 32), out.X().v * out.Y().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData ReorderWeightsBinaryKernel::GetKernelsData(const Params& params, const optional_params& options) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams, options);
}

KernelsPriority ReorderWeightsBinaryKernel::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_4;
}
}  // namespace kernel_selector
