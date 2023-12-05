// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_kernel_selector.h"
#include "group_normalization_kernel_ref.h"

namespace kernel_selector {

group_normalization_kernel_selector::group_normalization_kernel_selector() {
    Attach<GroupNormalizationKernelRef>();
}

KernelsData group_normalization_kernel_selector::GetBestKernels(const Params &params,
                                                                const optional_params &options) const {
    return GetNaiveBestKernel(params, options, KernelType::GROUP_NORMALIZATION);
}

}  // namespace kernel_selector
