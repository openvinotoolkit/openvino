// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_kernel_base.h"

namespace kernel_selector {
class LSTMCellKernelRef : public LSTMKernelBase {
public:
    LSTMCellKernelRef() : LSTMKernelBase("lstm_seq_ref") {}
    virtual ~LSTMCellKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
