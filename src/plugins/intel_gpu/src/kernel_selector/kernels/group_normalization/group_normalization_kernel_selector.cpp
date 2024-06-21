// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_kernel_selector.h"
#include "group_normalization_kernel_ref.h"
#include "group_normalization_kernel_bfyx_opt.h"

namespace kernel_selector {

group_normalization_kernel_selector::group_normalization_kernel_selector() {
    // Attach<GroupNormalizationKernelRef>();
    Attach<GroupNormalizationKernelBfyxOpt>();
}

KernelsData group_normalization_kernel_selector::GetBestKernels(const Params &params) const {
    return GetNaiveBestKernel(params, KernelType::GROUP_NORMALIZATION);
}

}  // namespace kernel_selector
