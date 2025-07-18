// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "sparse_fill_empty_rows_kernel_selector.h"
#include "sparse_fill_empty_rows_kernel_ref.h"

namespace kernel_selector {

sparse_fill_empty_rows_kernel_selector::sparse_fill_empty_rows_kernel_selector() {
    Attach<SparseFillEmptyRowsKernelRef>();
}

KernelsData sparse_fill_empty_rows_kernel_selector::GetBestKernels(const Params &params) const {
    return GetNaiveBestKernel(params, KernelType::SPARSE_FILL_EMPTY_ROWS);
}

} // namespace kernel_selector
