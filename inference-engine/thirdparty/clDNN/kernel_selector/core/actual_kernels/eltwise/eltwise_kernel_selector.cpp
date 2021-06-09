// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_selector.h"
#include "eltwise_kernel_ref.h"
#include "eltwise_kernel_vload8.h"
#include "eltwise_kernel_fs_b_yx_fsv32.h"
#include "eltwise_kernel_b_fs_yx_fsv16.h"
#include "eltwise_kernel_mixed_byxf_and_fs_b_yx_fsv32.h"
#include "eltwise_kernel_b_fs_yx_fsv4.h"

namespace kernel_selector {
eltwise_kernel_selector::eltwise_kernel_selector() {
    Attach<EltwiseKernelRef>();
    Attach<EltwiseKernel_vload8>();
    Attach<EltwiseKernel_fs_b_yx_fsv32>();
    Attach<EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32>();
    Attach<EltwiseKernel_b_fs_yx_fsv16>();
    Attach<EltwiseKernel_b_fs_yx_fsv4>();
}

KernelsData eltwise_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::ELTWISE);
}
}  // namespace kernel_selector
