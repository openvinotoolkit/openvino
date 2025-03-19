// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_seq_cm_kernel_selector.h"
#include "lstm_seq_cm.h"


namespace kernel_selector {
lstm_seq_cm_kernel_selector::lstm_seq_cm_kernel_selector() {
    Attach<LSTMSeqKernel_CM>();
}

KernelsData lstm_seq_cm_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::LSTM_SEQ_CELL);
}
}  // namespace kernel_selector
