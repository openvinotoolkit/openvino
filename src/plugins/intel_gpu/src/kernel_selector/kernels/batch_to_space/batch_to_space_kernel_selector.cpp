// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_to_space_kernel_selector.h"
#include "batch_to_space_kernel_ref.h"

namespace kernel_selector {

batch_to_space_kernel_selector::batch_to_space_kernel_selector() {
    Attach<BatchToSpaceKernelRef>();
}

KernelsData batch_to_space_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::BATCH_TO_SPACE);
}
}  // namespace kernel_selector
