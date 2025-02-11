// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "search_sorted_kernel_selector.h"
#include "search_sorted_kernel_ref.h"

namespace kernel_selector {
search_sorted_kernel_selector::search_sorted_kernel_selector() { Attach<SearchSortedKernelRef>(); }

KernelsData search_sorted_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SEARCH_SORTED);
}
}  // namespace kernel_selector
