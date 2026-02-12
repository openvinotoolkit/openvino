// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "segment_max_kernel_selector.h"
#include "segment_max_kernel_ref.h"
#include "segment_max_kernel_opt.h"

namespace kernel_selector {

segment_max_kernel_selector::segment_max_kernel_selector() {
    Attach<SegmentMaxKernelRef>();
    Attach<SegmentMaxKernelOpt>();
}

KernelsData segment_max_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SEGMENT_MAX);
}
}  // namespace kernel_selector
