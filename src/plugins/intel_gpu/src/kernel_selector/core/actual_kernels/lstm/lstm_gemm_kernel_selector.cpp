// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_gemm_kernel_selector.h"
#include "lstm_gemm_kernel_ref.h"
#include "lstm_gemv_gpu_subgroup1x64_bfyx_ff_simd16.h"
#include "lstm_gemv_gpu_subgroup1x64_bfyx_hh_simd16.h"

namespace kernel_selector {
lstm_gemm_kernel_selector::lstm_gemm_kernel_selector() {
    Attach<LSTMGemmKernelRef>();
    Attach<LSTMGemvKernel_subgroup1x64_bfyx_ff_SIMD16>();
    Attach<LSTMGemvKernel_subgroup1x64_bfyx_hh_SIMD16>();
}

KernelsData lstm_gemm_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::LSTM_GEMM);
}
}  // namespace kernel_selector