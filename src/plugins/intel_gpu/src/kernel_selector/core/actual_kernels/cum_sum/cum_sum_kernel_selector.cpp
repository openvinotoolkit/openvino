// Copyright (C) 2018-2022 Intel Corporation
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

KernelsData cum_sum_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::CUM_SUM);
}
}  // namespace kernel_selector
