// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "count_nonzero_kernel_selector.h"
#include "count_nonzero_kernel_ref.h"

namespace kernel_selector {

count_nonzero_kernel_selector::count_nonzero_kernel_selector() { Attach<CountNonzeroKernelRef>(); }

KernelsData count_nonzero_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::COUNT_NONZERO);
}
}  // namespace kernel_selector
