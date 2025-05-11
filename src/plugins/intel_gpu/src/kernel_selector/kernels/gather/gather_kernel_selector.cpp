// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_kernel_selector.h"
#include "gather_kernel_ref.h"

namespace kernel_selector {

gather_kernel_selector::gather_kernel_selector() { Attach<GatherKernelRef>(); }

KernelsData gather_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::GATHER);
}
}  // namespace kernel_selector
