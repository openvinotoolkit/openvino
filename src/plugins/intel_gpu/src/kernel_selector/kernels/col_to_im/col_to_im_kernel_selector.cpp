// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col_to_im_kernel_selector.h"
#include "col_to_im_kernel_ref.h"
#include "col_to_im_kernel_opt.h"

namespace kernel_selector {

col_to_im_kernel_selector::col_to_im_kernel_selector() {
    Attach<ColToImKernelRef>();
    Attach<ColToImKernelOpt>();
}

KernelsData col_to_im_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::COL_TO_IM);
}
}  // namespace kernel_selector
