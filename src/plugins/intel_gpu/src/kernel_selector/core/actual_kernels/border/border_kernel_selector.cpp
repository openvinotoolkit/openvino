// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "border_kernel_selector.h"
#include "border_kernel_ref.h"

namespace kernel_selector {
border_kernel_selector::border_kernel_selector() { Attach<BorderKernelRef>(); }

KernelsData border_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::BORDER);
}
}  // namespace kernel_selector
