// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col2im_kernel_selector.h"
#include "col2im_kernel_ref.h"
#include "col2im_kernel_opt.h"

namespace kernel_selector {

col2im_kernel_selector::col2im_kernel_selector() {
    Attach<Col2ImKernelRef>();
    Attach<Col2ImKernelOpt>();
}

KernelsData col2im_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::COL2IM);
}
}  // namespace kernel_selector
