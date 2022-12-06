// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reverse_kernel_selector.h"

#include "reverse_kernel_ref.h"

namespace kernel_selector {

reverse_kernel_selector::reverse_kernel_selector() {
    Attach<ReverseKernelRef>();
}

KernelsData reverse_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::REVERSE);
}
}  // namespace kernel_selector
