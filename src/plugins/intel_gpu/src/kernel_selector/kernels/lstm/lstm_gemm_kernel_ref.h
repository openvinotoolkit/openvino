// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_gemm_kernel_base.h"

namespace kernel_selector {
class LSTMGemmKernelRef : public LSTMGemmKernelBase {
public:
    LSTMGemmKernelRef() : LSTMGemmKernelBase("lstm_gemm_gpu_bfyx_ref") {}
    virtual ~LSTMGemmKernelRef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
