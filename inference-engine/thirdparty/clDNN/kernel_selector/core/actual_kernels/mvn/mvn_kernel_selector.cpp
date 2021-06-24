// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_kernel_selector.h"
#include "mvn_kernel_ref.h"
#include "mvn_kernel_bfyx_opt.h"
#include "mvn_kernel_b_fs_yx_fsv16_imad.hpp"

namespace kernel_selector {
mvn_kernel_selector::mvn_kernel_selector() {
    Attach<MVNKernelRef>();
    Attach<MVNKernelBfyxOpt>();
    Attach<MVNKernel_b_fs_yx_fsv16_imad>();
}

KernelsData mvn_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::MVN);
}
}  // namespace kernel_selector
