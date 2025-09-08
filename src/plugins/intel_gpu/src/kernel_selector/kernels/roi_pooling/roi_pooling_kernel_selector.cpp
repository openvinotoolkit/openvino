// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roi_pooling_kernel_selector.h"
#include "roi_pooling_kernel_ref.h"
#include "roi_pooling_kernel_ps_ref.h"

namespace kernel_selector {
roi_pooling_kernel_selector::roi_pooling_kernel_selector() {
    Attach<ROIPoolingKernelRef>();
    Attach<PSROIPoolingKernelRef>();
}

KernelsData roi_pooling_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::ROI_POOLING);
}
}  // namespace kernel_selector
