// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bevpool_v2_kernel_selector.h"

#include "bevpool_v2_kernel_ref.h"

namespace kernel_selector {

bevpool_v2_kernel_selector::bevpool_v2_kernel_selector() {
    Attach<BevPoolV2KernelRef>();
}

KernelsData bevpool_v2_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::BEVPOOL_V2);
}

}  // namespace kernel_selector
