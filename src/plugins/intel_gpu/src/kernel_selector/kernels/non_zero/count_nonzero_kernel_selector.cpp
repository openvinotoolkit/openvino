// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "count_nonzero_kernel_selector.h"
#include "count_nonzero_kernel_ref.h"

namespace kernel_selector {

count_nonzero_kernel_selector::count_nonzero_kernel_selector() { Attach<CountNonzeroKernelRef>(); }

KernelsData count_nonzero_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::COUNT_NONZERO);
}
}  // namespace kernel_selector
