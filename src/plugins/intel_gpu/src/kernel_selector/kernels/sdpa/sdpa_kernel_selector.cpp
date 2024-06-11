// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_selector.h"
#include "sdpa_kernel_ref.h"
#include "sdpa_kernel_opt.h"

namespace kernel_selector {

sdpa_kernel_selector::sdpa_kernel_selector() {
    Attach<SDPAKernelOpt>();
    Attach<SDPAKernelRef>();
}

KernelsData sdpa_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SDPA);
}
}  // namespace kernel_selector
