// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_dynamic_timeloop_kernel_selector.h"
#include "lstm_dynamic_timeloop_ref_kernel.h"

namespace kernel_selector {
lstm_dynamic_timeloop_kernel_selector::lstm_dynamic_timeloop_kernel_selector() {
    Attach<LSTM_DynamicTimeloopKernelRef>();
}

KernelsData lstm_dynamic_timeloop_kernel_selector::GetBestKernels(const Params& params,
                                                                  const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::LSTM_DYNAMIC_TIMELOOP);
}
}  // namespace kernel_selector
