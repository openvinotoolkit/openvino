// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_cell_and_seq_kernel_selector.h"
#include "lstm_cell_and_seq_kernel_ref.h"
#include "lstm_cell_and_seq_kernel_bfyx.h"

namespace kernel_selector {
lstm_cell_and_seq_kernel_selector::lstm_cell_and_seq_kernel_selector() {
    Attach<LSTMCellAndSeqKernelRef>();
    Attach<LSTMCellAndSeqKernel_bfyx>();
}

KernelsData lstm_cell_and_seq_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::LSTM_SEQ_CELL);
}
}  // namespace kernel_selector
