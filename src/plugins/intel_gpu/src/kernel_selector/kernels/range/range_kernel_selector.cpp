// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "range_kernel_selector.h"
#include "range_kernel_ref.h"

namespace kernel_selector {

range_kernel_selector::range_kernel_selector() {
    Attach<RangeKernelRef>();
}

KernelsData range_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::RANGE);
}
}  // namespace kernel_selector
