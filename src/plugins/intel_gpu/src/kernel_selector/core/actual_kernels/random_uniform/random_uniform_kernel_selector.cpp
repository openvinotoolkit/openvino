// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "random_uniform_kernel_selector.h"
#include "random_uniform_kernel_ref.h"

namespace kernel_selector {

random_uniform_kernel_selector::random_uniform_kernel_selector() {
    Attach<RandomUniformKernelRef>();
}

KernelsData random_uniform_kernel_selector::GetBestKernels(const Params &params, const optional_params &options) const {
    return GetNaiveBestKernel(params, options, KernelType::RANDOM_UNIFORM);
}
} // namespace kernel_selector
