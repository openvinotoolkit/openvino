// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unique_kernel_selector.hpp"

#include "unique_kernel_ref.hpp"

namespace kernel_selector {

unique_kernel_selector::unique_kernel_selector() {
    Attach<UniqueKernelRef>();
}

KernelsData unique_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::UNIQUE);
}

unique_kernel_selector& unique_kernel_selector::Instance() {
    static unique_kernel_selector instance;
    return instance;
}

unique_reshape_kernel_selector::unique_reshape_kernel_selector() {
    Attach<UniqueReshapeKernelRef>();
}

KernelsData unique_reshape_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::UNIQUE_RESHAPE);
}

unique_reshape_kernel_selector& unique_reshape_kernel_selector::Instance() {
    static unique_reshape_kernel_selector instance;
    return instance;
}

}  // namespace kernel_selector
