// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adaptive_pooling_kernel_selector.h"
#include "adaptive_pooling_kernel_ref.h"

namespace kernel_selector {
adaptive_pooling_kernel_selector::adaptive_pooling_kernel_selector() {
    Attach<AdaptivePoolingRef>();
}

adaptive_pooling_kernel_selector& adaptive_pooling_kernel_selector::Instance() {
    static adaptive_pooling_kernel_selector instance_;
    return instance_;
}

KernelsData adaptive_pooling_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::ADAPTIVE_POOLING);
}
}  // namespace kernel_selector
