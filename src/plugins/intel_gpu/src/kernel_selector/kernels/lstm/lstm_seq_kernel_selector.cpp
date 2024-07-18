// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_seq_kernel_selector.h"
#include "lstm_seq_kernel_ref.h"

namespace kernel_selector {
lstm_seq_kernel_selector::lstm_seq_kernel_selector() { Attach<LSTMSeqKernelRef>(); }

KernelsData lstm_seq_kernel_selector::GetBestKernels(const Params& params) const {
    auto k = GetNaiveBestKernel(params, KernelType::LSTM_SEQ);
    return k;
}
}  // namespace kernel_selector
