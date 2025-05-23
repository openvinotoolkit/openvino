// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_kernel_base.h"

namespace kernel_selector {
class LSTMCellAndSeqKernelRef : public LSTMKernelBase {
public:
    LSTMCellAndSeqKernelRef() : LSTMKernelBase("lstm_cell_and_seq_ref") {}
    virtual ~LSTMCellAndSeqKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
