// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_update_kernel_selector.h"
#include "scatter_update_kernel_ref.h"

namespace kernel_selector {

scatter_update_kernel_selector::scatter_update_kernel_selector() { Attach<ScatterUpdateKernelRef>(); }

KernelsData scatter_update_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::SCATTER_UPDATE);
}
}  // namespace kernel_selector
