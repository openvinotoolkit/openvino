// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_kernel_selector.h"
#include "activation_kernel_opt.h"
#include "activation_kernel_ref.h"

namespace kernel_selector {
activation_kernel_selector::activation_kernel_selector() {
    Attach<ActivationKernelRef>();
    Attach<ActivationKernelOpt>();
}

KernelsData activation_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::ACTIVATION);
}
}  // namespace kernel_selector
