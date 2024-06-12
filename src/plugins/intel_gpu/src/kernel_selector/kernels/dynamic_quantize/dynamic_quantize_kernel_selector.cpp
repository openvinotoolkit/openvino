// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_quantize_kernel_selector.h"
#include "dynamic_quantize_kernel_ref.h"

namespace kernel_selector {
dynamic_quantize_kernel_selector::dynamic_quantize_kernel_selector() { Attach<DynamicQuantizeKernelRef>(); }

KernelsData dynamic_quantize_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::DYNAMIC_QUANTIZE);
}
}  // namespace kernel_selector
