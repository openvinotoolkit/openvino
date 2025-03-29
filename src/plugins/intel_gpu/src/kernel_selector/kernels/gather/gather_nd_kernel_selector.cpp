// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_nd_kernel_selector.h"
#include "gather_nd_kernel_ref.h"

namespace kernel_selector {

gather_nd_kernel_selector::gather_nd_kernel_selector() { Attach<GatherNDKernelRef>(); }

KernelsData gather_nd_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::GATHER_ND);
}
}  // namespace kernel_selector
