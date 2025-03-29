// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "roi_align_kernel_selector.h"
#include "roi_align_kernel_ref.h"

namespace kernel_selector {

roi_align_kernel_selector::roi_align_kernel_selector() {
    Attach<ROIAlignKernelRef>();
}

KernelsData roi_align_kernel_selector::GetBestKernels(const Params &params) const {
    return GetNaiveBestKernel(params, KernelType::ROI_ALIGN);
}

} // namespace kernel_selector
