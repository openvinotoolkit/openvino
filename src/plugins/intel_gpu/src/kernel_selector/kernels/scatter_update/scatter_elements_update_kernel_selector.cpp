// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_elements_update_kernel_selector.h"
#include "scatter_elements_update_kernel_ref.h"

namespace kernel_selector {

scatter_elements_update_kernel_selector::scatter_elements_update_kernel_selector() { Attach<ScatterElementsUpdateKernelRef>(); }

KernelsData scatter_elements_update_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::SCATTER_ELEMENTS_UPDATE);
}
}  // namespace kernel_selector
