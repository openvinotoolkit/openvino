// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_nd_update_kernel_selector.h"
#include "scatter_nd_update_kernel_ref.h"

namespace kernel_selector {

scatter_nd_update_kernel_selector::scatter_nd_update_kernel_selector() { Attach<ScatterNDUpdateKernelRef>(); }

KernelsData scatter_nd_update_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SCATTER_ND_UPDATE);
}
}  // namespace kernel_selector
