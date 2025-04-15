// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_color_kernel_selector.h"
#include "convert_color_kernel_ref.h"

namespace kernel_selector {

convert_color_kernel_selector::convert_color_kernel_selector() {
    Attach<ConvertColorKernelRef>();
}

KernelsData convert_color_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::CONVERT_COLOR);
}
}  // namespace kernel_selector
