// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice_kernel_selector.h"
#include "strided_slice_kernel_ref.h"

namespace kernel_selector {

strided_slice_kernel_selector::strided_slice_kernel_selector() { Attach<StridedSliceKernelRef>(); }

KernelsData strided_slice_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::STRIDED_SLICE);
}
}  // namespace kernel_selector
