// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swiglu_kernel_selector.h"
#include "swiglu_kernel_ref.h"

namespace kernel_selector {
swiglu_kernel_selector::swiglu_kernel_selector() { Attach<SwiGLUKernelRef>(); }

KernelsData swiglu_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SWIGLU);
}
}  // namespace kernel_selector
