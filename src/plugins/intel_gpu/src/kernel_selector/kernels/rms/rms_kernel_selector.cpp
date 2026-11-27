// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_kernel_selector.h"
#include "rms_kernel_ref.h"
#include "rms_kernel_bfyx_opt.h"
#include "rms_kernel_fsv16.h"

namespace kernel_selector {
rms_kernel_selector::rms_kernel_selector() {
    Attach<RMSKernelRef>();
    Attach<RMSKernelBfyxOpt>();
    Attach<RMSKernelFsv16>();
}

KernelsData rms_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::RMS);
}
}  // namespace kernel_selector
