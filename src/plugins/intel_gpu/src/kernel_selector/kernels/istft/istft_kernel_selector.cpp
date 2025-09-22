// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "istft_kernel_selector.h"

#include "istft_kernel_ref.h"

namespace kernel_selector {
ISTFT_kernel_selector::ISTFT_kernel_selector() {
    Attach<ISTFTKernelRef>();
}

KernelsData ISTFT_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::ISTFT);
}
}  // namespace kernel_selector
