// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "border_kernel_selector.h"
#include "border_kernel_ref.h"

namespace kernel_selector {
border_kernel_selector::border_kernel_selector() { Attach<BorderKernelRef>(); }

KernelsData border_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::BORDER);
}
}  // namespace kernel_selector
