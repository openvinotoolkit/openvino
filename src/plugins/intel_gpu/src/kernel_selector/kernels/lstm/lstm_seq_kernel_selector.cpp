// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_seq_kernel_selector.h"
#include "lstm_seq_kernel_ref.h"
#include "lstm_seq_kernel_bfyx.h"

namespace kernel_selector {
lstm_seq_kernel_selector::lstm_seq_kernel_selector() {
    Attach<LSTMSeqKernelRef>();
    Attach<LSTMSeqKernel_bfyx>();
}

KernelsData lstm_seq_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::LSTM_SEQ_CELL);
}
}  // namespace kernel_selector
