// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_kernel_selector.h"
#include "rms_kernel_ref.h"
#include "rms_kernel_bfyx_opt.h"

namespace kernel_selector {
rms_kernel_selector::rms_kernel_selector() {
    Attach<RMSKernelRef>();
    Attach<RMSKernelBfyxOpt>();
}

KernelsData rms_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::RMS);
}
}  // namespace kernel_selector
