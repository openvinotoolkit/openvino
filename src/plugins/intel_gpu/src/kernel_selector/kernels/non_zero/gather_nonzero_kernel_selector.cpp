// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_nonzero_kernel_selector.h"
#include "gather_nonzero_kernel_ref.h"

namespace kernel_selector {

gather_nonzero_kernel_selector::gather_nonzero_kernel_selector() { Attach<GatherNonzeroKernelRef>(); }

KernelsData gather_nonzero_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::GATHER_NONZERO);
}
}  // namespace kernel_selector
