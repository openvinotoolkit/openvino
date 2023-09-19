// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_kernel_selector.h"
#include "mha_kernel_ref.h"
#include "mha_kernel_opt.h"

namespace kernel_selector {

mha_kernel_selector::mha_kernel_selector() {
    // Attach<MHAKernelRef>();
    Attach<MHAKernelOpt>();
}

KernelsData mha_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::MHA);
}
}  // namespace kernel_selector
