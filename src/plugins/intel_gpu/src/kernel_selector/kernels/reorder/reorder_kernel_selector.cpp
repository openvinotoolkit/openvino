// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel_selector.h"
#include "reorder_kernel.h"
#include "reorder_kernel_fast_b1.h"
#include "reorder_from_winograd_2x3_kernel.h"
#include "reorder_to_winograd_2x3_kernel.h"
#include "reorder_kernel_to_yxfb_batched.h"
#include "reorder_biplanar_nv12.h"
#include "reorder_kernel_fs_b_yx_fsv32_to_bfyx.h"
#include "reorder_kernel_bfyx_to_blocked_format.h"
#include "reorder_kernel_b_fs_yx_fsv16_fsv32_to_bfyx.h"

namespace kernel_selector {

reorder_kernel_selector::reorder_kernel_selector() {
    Attach<ReorderKernelRef>();
    Attach<ReorderKernelFastBatch1>();
    Attach<ReorderFromWinograd2x3Kernel>();
    Attach<ReorderToWinograd2x3Kernel>();
    Attach<ReorderKernel_to_yxfb_batched>();
    Attach<reorder_biplanar_nv12>();
    Attach<ReorderKernel_fs_b_yx_fsv32_to_bfyx>();
    Attach<ReorderKernel_bfyx_to_blocked_format>();
    Attach<ReorderKernel_b_fs_yx_fsv16_fsv32_to_bfyx>();
}

KernelsData reorder_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::REORDER);
}
}  // namespace kernel_selector
