// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "search_sorted_kernel_base.h"

namespace kernel_selector {
class SearchSortedKernelRef : public SearchSortedKernelBase {
public:
    SearchSortedKernelRef() : SearchSortedKernelBase("search_sorted_ref") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
