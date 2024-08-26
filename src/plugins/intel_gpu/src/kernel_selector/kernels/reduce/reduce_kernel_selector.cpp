// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_kernel_selector.h"
#include "reduce_kernel_ref.h"
#include "reduce_kernel_b_fs_yx_fsv16.h"
#include "reduce_kernel_simple_to_scalar.h"

namespace kernel_selector {

reduce_kernel_selector::reduce_kernel_selector() {
    Attach<ReduceKernelRef>();
    Attach<ReduceKernel_b_fs_yx_fsv16>();
    Attach<ReduceKernelSimpleToScalar>();
}

KernelsData reduce_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::REDUCE);
}
}  // namespace kernel_selector
