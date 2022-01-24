// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyramid_roi_align_kernel_selector.h"
#include "pyramid_roi_align_kernel_ref.h"

namespace kernel_selector {
PyramidROIAlign_kernel_selector::PyramidROIAlign_kernel_selector() { Attach<PyramidROIAlignKernelRef>(); }

KernelsData PyramidROIAlign_kernel_selector::GetBestKernels(const Params& params,
                                                            const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::PYRAMID_ROI_ALIGN);
}
}  // namespace kernel_selector