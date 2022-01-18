// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_kernel_selector.h"
#include "broadcast_kernel_ref.h"

namespace kernel_selector {
broadcast_kernel_selector::broadcast_kernel_selector() { Attach<BroadcastKernelRef>(); }

KernelsData broadcast_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::BROADCAST);
}
}  // namespace kernel_selector
