// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_gemm_kernel_base.h"

namespace kernel_selector {
class LSTMGemvKernel_subgroup1x64_bfyx_hh_SIMD16 : public LSTMGemmKernelBase {
public:
    LSTMGemvKernel_subgroup1x64_bfyx_hh_SIMD16() : LSTMGemmKernelBase("lstm_gemv_gpu_subgroup1x64_bfyx_hh_SIMD16") {}
    virtual ~LSTMGemvKernel_subgroup1x64_bfyx_hh_SIMD16() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
