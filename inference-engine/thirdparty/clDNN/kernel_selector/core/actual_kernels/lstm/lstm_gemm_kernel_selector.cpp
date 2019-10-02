/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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