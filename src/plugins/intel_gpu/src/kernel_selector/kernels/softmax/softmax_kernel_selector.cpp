// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_kernel_selector.h"
#include "softmax_kernel_ref.h"
#include "softmax_kernel_bf.h"
#include "softmax_kernel_fb.h"
#include "softmax_kernel_items_class_optimized.h"

namespace kernel_selector {

softmax_kernel_selector::softmax_kernel_selector() {
    Attach<SoftmaxKernelRef>();
    Attach<SoftmaxKernel_bf>();
    Attach<SoftmaxKernel_fb>();
    Attach<SoftmaxKerneItemsClassOptimized>();
}

KernelsData softmax_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SOFT_MAX);
}
}  // namespace kernel_selector
