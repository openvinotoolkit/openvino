// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_dynamic_input_kernel_selector.h"
#include "lstm_dynamic_input_ref_kernel.h"
#include "lstm_dynamic_input_bfyx_opt.h"

namespace kernel_selector {
lstm_dynamic_input_kernel_selector::lstm_dynamic_input_kernel_selector() {
    Attach<LSTM_DynamicInputKernelRef>();
    Attach<LSTM_DynamicInputKernelBfyxOpt>();
}

KernelsData lstm_dynamic_input_kernel_selector::GetBestKernels(const Params& params,
                                                               const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::LSTM_DYNAMIC_INPUT);
}
}  // namespace kernel_selector
