// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "feed_forward_kernel_selector.h"
#include "feed_forward_kernel_ref.h"

namespace kernel_selector {
feed_forward_kernel_selector::feed_forward_kernel_selector() {
    Attach<FeedForwardKernelRef>();
}

KernelsData feed_forward_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::FEED_FORWARD);
}
}  // namespace kernel_selector
