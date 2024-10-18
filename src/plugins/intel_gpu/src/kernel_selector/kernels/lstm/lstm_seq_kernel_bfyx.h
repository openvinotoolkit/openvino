// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_kernel_base.h"

namespace kernel_selector {
class LSTMSeqKernel_bfyx : public LSTMKernelBase {
public:
    LSTMSeqKernel_bfyx() : LSTMKernelBase("lstm_cell_and_seq_bfyx") {}
    virtual ~LSTMSeqKernel_bfyx() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& p) const override;
};
}  // namespace kernel_selector
