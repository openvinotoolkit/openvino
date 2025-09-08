// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cum_sum_kernel_selector.h"
#include "cum_sum_kernel_ref.h"
#include "cum_sum_kernel_partial_sum.h"

namespace kernel_selector {
cum_sum_kernel_selector::cum_sum_kernel_selector() {
    Attach<CumSumKernelRef>();
    Attach<CumSumKernelPartialSum>();
}

KernelsData cum_sum_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::CUM_SUM);
}
}  // namespace kernel_selector
