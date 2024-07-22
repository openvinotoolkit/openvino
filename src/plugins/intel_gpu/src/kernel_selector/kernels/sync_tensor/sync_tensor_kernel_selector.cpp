// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sync_tensor_kernel_selector.h"
#include "sync_tensor_kernel_ref.h"

namespace kernel_selector {
sync_tensor_kernel_selector::sync_tensor_kernel_selector() {
    std::cout << "[kernel] sync_tensor_kernel_selector" << std::endl;
    Attach<SyncTensorKernelRef>();
}

KernelsData sync_tensor_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SYNC_TENSOR);
}
}  // namespace kernel_selector
