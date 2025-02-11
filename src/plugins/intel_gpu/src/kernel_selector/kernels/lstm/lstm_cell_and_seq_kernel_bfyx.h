// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_kernel_base.h"

namespace kernel_selector {
class LSTMCellAndSeqKernel_bfyx : public LSTMKernelBase {
public:
    LSTMCellAndSeqKernel_bfyx() : LSTMKernelBase("lstm_cell_and_seq_bfyx") {}
    virtual ~LSTMCellAndSeqKernel_bfyx() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
