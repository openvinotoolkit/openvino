// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_cm_kernel_selector.h"

#include "fully_connected_cm_example.h"

namespace kernel_selector {
fully_connected_cm_kernel_selector::fully_connected_cm_kernel_selector() {
    Attach<FullyConnected_cm_example>();
}

KernelsData fully_connected_cm_kernel_selector::GetBestKernels(const Params& params) const {
    return GetAutoTuneBestKernel(params, KernelType::FULLY_CONNECTED);
}
}  // namespace kernel_selector
