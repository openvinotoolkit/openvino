// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "multinomial_kernel_selector.h"
#include "multinomial_kernel_ref.h"

namespace kernel_selector {

multinomial_kernel_selector::multinomial_kernel_selector() {
    Attach<MultinomialKernelRef>();
}

KernelsData multinomial_kernel_selector::GetBestKernels(const Params &params,
                                                                const optional_params &options) const {
    return GetNaiveBestKernel(params, options, KernelType::MULTINOMIAL);
}

}  // namespace kernel_selector
