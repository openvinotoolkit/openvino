// Copyright (C) 2018-2022 Intel Corporation
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

KernelsData activation_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::ACTIVATION);
}
}  // namespace kernel_selector
