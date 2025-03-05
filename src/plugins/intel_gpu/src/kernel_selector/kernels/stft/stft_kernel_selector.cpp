// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stft_kernel_selector.h"

#include "stft_kernel_opt.h"
#include "stft_kernel_ref.h"

namespace kernel_selector {
STFT_kernel_selector::STFT_kernel_selector() {
    Attach<STFTKernelRef>();
    Attach<STFTKernelOpt>();
}

KernelsData STFT_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::STFT);
}
}  // namespace kernel_selector
