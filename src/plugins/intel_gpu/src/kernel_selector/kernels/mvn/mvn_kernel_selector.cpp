// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_kernel_selector.h"
#include "mvn_kernel_ref.h"
#include "mvn_kernel_bfyx_opt.h"
#include "mvn_kernel_b_fs_yx_fsv16_imad.hpp"
#include "mvn_kernel_bs_fs_yx_bsv32.hpp"

namespace kernel_selector {
mvn_kernel_selector::mvn_kernel_selector() {
    Attach<MVNKernelRef>();
    Attach<MVNKernelBfyxOpt>();
    Attach<MVNKernel_b_fs_yx_fsv16_imad>();
    Attach<MVNKernel_bs_fs_yx_bsv32>();
}

KernelsData mvn_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::MVN);
}
}  // namespace kernel_selector
