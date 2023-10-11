// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unique_kernel_selector.hpp"

#include "unique_kernel_ref.hpp"

namespace kernel_selector {

unique_count_kernel_selector::unique_count_kernel_selector() {
    Attach<UniqueCountKernelRef>();
}

KernelsData unique_count_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::UNIQUE_COUNT);
}

unique_count_kernel_selector& unique_count_kernel_selector::Instance() {
    static unique_count_kernel_selector instance;
    return instance;
}

unique_gather_kernel_selector::unique_gather_kernel_selector() {
    Attach<UniqueGatherKernelRef>();
}

KernelsData unique_gather_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::UNIQUE_GATHER);
}

unique_gather_kernel_selector& unique_gather_kernel_selector::Instance() {
    static unique_gather_kernel_selector instance;
    return instance;
}

}  // namespace kernel_selector
