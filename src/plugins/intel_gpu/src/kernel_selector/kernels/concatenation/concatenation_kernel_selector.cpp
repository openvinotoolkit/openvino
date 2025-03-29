// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concatenation_kernel_selector.h"
#include "concatenation_kernel_ref.h"
#include "concatenation_kernel_simple_ref.h"
#include "concatenation_kernel_depth_bfyx_no_pitch.h"
#include "concatenation_kernel_b_fs_yx_fsv16.h"
#include "concatenation_kernel_fs_b_yx_fsv32.h"

namespace kernel_selector {
concatenation_kernel_selector::concatenation_kernel_selector() {
    Attach<ConcatenationKernelRef>();
    Attach<ConcatenationKernel_simple_Ref>();
    Attach<ConcatenationKernel_depth_bfyx_no_pitch>();
    Attach<ConcatenationKernel_b_fs_yx_fsv16>();
    Attach<ConcatenationKernel_fs_b_yx_fsv32>();
}

KernelsData concatenation_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::CONCATENATION);
}
}  // namespace kernel_selector
