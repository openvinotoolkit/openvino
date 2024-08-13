// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_cell_kernel_selector.h"
#include "lstm_cell_kernel_ref.h"

namespace kernel_selector {
lstm_cell_kernel_selector::lstm_cell_kernel_selector() { Attach<LSTMCellKernelRef>(); }

KernelsData lstm_cell_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::LSTM_CELL);
}
}  // namespace kernel_selector
