// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "depth_to_space_kernel_selector.h"
#include "depth_to_space_kernel_ref.h"
#include "depth_to_space_kernel_block2_opt.h"

namespace kernel_selector {

depth_to_space_kernel_selector::depth_to_space_kernel_selector() {
    Attach<DepthToSpaceKernelRef>();
    Attach<DepthToSpaceKernelBlock2Opt>();
}

KernelsData depth_to_space_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::DEPTH_TO_SPACE);
}
}  // namespace kernel_selector
