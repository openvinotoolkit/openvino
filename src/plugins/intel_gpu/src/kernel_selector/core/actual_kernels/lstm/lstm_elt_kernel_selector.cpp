// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_elt_kernel_selector.h"
#include "lstm_elt_kernel_ref.h"

namespace kernel_selector {
lstm_elt_kernel_selector::lstm_elt_kernel_selector() { Attach<LSTMEltKernelRef>(); }

KernelsData lstm_elt_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::LSTM_ELT);
}
}  // namespace kernel_selector