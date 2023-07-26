// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_kernel_selector.h"

namespace kernel_selector {

kernel_selector::group_normalization_kernel_selector::group_normalization_kernel_selector() {
}

KernelsData kernel_selector::group_normalization_kernel_selector::GetBestKernels(const Params &params,
                                                                                 const optional_params &options) const {
    return GetNaiveBestKernel(params, options, KernelType::GROUP_NORMALIZATION);
}

}  // namespace kernel_selector
