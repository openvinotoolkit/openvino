// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_elements_update_kernel_selector.h"
#include "scatter_elements_update_kernel_ref.h"

namespace kernel_selector {

scatter_elements_update_kernel_selector::scatter_elements_update_kernel_selector() { Attach<ScatterElementsUpdateKernelRef>(); }

KernelsData scatter_elements_update_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SCATTER_ELEMENTS_UPDATE);
}
}  // namespace kernel_selector
