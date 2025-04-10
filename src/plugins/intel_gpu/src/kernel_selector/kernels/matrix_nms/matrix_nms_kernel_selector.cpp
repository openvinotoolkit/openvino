// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matrix_nms_kernel_selector.h"

#include "matrix_nms_kernel_ref.h"

namespace kernel_selector {

matrix_nms_kernel_selector::matrix_nms_kernel_selector() {
    Attach<MatrixNmsKernelRef>();
}

KernelsData matrix_nms_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::MATRIX_NMS);
}
}  // namespace kernel_selector
