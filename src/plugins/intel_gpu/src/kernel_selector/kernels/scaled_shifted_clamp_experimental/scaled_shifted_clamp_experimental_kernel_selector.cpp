// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_shifted_clamp_experimental_kernel_selector.h"

#include "scaled_shifted_clamp_experimental_kernel_ref.h"

namespace kernel_selector {

scaled_shifted_clamp_experimental_kernel_selector::scaled_shifted_clamp_experimental_kernel_selector() {
    Attach<ScaledShiftedClampExperimentalKernelRef>();
}

KernelsData scaled_shifted_clamp_experimental_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SCALED_SHIFTED_CLAMP_EXPERIMENTAL);
}

}  // namespace kernel_selector
