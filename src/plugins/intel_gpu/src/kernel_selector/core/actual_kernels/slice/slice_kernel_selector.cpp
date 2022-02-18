// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "slice_kernel_selector.h"
#include "slice_kernel_ref.h"

namespace kernel_selector {

slice_kernel_selector::slice_kernel_selector() {
    Attach<SliceKernelRef>();
}

KernelsData slice_kernel_selector::GetBestKernels(const Params &params,
        const optional_params &options) const {
    return GetNaiveBestKernel(params, options, KernelType::SLICE);
}

} // namespace kernel_selector
