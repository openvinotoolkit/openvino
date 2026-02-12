// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "slice_scatter_kernel_selector.h"
#include "slice_scatter_kernel_ref.h"
#include "slice_scatter_kernel_opt.h"

namespace kernel_selector {

slice_scatter_kernel_selector::slice_scatter_kernel_selector() {
    Attach<SliceScatterKernelRef>();
    Attach<SliceScatterKernelOpt>();
}

KernelsData slice_scatter_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SLICE_SCATTER);
}

}  // namespace kernel_selector
