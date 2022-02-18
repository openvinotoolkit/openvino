// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_kernel_selector.h"
#include "reduce_kernel_ref.h"
#include "reduce_kernel_b_fs_yx_fsv16.h"

namespace kernel_selector {

reduce_kernel_selector::reduce_kernel_selector() {
    Attach<ReduceKernelRef>();
    Attach<ReduceKernel_b_fs_yx_fsv16>();
}

KernelsData reduce_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::REDUCE);
}
}  // namespace kernel_selector
