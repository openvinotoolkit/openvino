// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_of_kernel_selector.h"
#include "shape_of_kernel_ref.h"

namespace kernel_selector {

shape_of_kernel_selector::shape_of_kernel_selector() {
    Attach<ShapeOfKernelRef>();
}

KernelsData shape_of_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SHAPE_OF);
}
}  // namespace kernel_selector
