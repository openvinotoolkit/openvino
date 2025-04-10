// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_kernel_selector.h"
#include "reshape_kernel_ref.h"

namespace kernel_selector {

reshape_kernel_selector::reshape_kernel_selector() { Attach<ReshapeKernelRef>(); }

KernelsData reshape_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::RESHAPE);
}
}  // namespace kernel_selector
