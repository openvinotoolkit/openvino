// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample_kernel_opt_bilinear.hpp"

#include "kernel_selector_utils.h"

namespace kernel_selector {

CommonDispatchData GridSampleKernelOptBilinear::CalcDispatch(const grid_sample_params& kernel_params) const {
    CommonDispatchData dispatch_data;
    const auto& output = kernel_params.outputs.front();

    dispatch_data.gws = {output.Batch().v * output.Feature().v, output.Y().v, output.X().v};
    dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, kernel_params.engineInfo);

    return dispatch_data;
}

KernelsPriority GridSampleKernelOptBilinear::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}

}  // namespace kernel_selector
