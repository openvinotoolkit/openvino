// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_kernel_selector.h"
#include "rope_kernel_ref.h"

namespace kernel_selector {
rope_kernel_selector::rope_kernel_selector() {
    Attach<RoPEKernelRef>();
}

KernelsData rope_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::ROPE);
}
}  // namespace kernel_selector
