// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_cm.h"
#include "lstm/lstm_kernel_base.h"

namespace kernel_selector {
class LSTMSeqKernel_CM : public KernelBaseCM {
public:
    LSTMSeqKernel_CM() : KernelBaseCM("lstm_gemm_loop") {}
    virtual ~LSTMSeqKernel_CM() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& p) const override;
private:
    struct lstm_shape{
        size_t batch_size, input_size, hidden_size, seq_len, num_dir, num_gates;
    };
    lstm_shape GetShape(const Params& params) const;
};
}  // namespace kernel_selector
