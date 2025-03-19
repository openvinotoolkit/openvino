// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize_kernel_selector.h"

#include "normalize_kernel_across_spatial_ref.h"
#include "normalize_kernel_within_spatial_ref.h"

namespace kernel_selector {
normalize_kernel_selector::normalize_kernel_selector() {
    Attach<NormalizeKernelWithinSpatialRef>();
    Attach<NormalizeKernelAcrossSpatialRef>();
}

KernelsData normalize_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::NORMALIZE);
}
}  // namespace kernel_selector
