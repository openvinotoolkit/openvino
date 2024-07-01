// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_selector.h"
#include "sdpa_kernel_ref.h"
#include "sdpa_kernel_opt.h"
#include "sdpa_kernel_micro.h"

namespace kernel_selector {

sdpa_kernel_selector::sdpa_kernel_selector() {
    Attach<SDPAKernelOpt>();
    Attach<SDPAKernelRef>();
#ifdef ENABLE_ONEDNN_FOR_GPU
    Attach<SDPAKernelMicro>();
#endif
}

KernelsData sdpa_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SDPA);
}
}  // namespace kernel_selector
