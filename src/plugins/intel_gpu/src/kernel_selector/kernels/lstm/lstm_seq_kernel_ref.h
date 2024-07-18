// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_seq_kernel_base.h"

namespace kernel_selector {
class LSTMSeqKernelRef : public LSTMSeqKernelBase {
public:
    LSTMSeqKernelRef() : LSTMSeqKernelBase("lstm_seq_gpu_bfyx_ref") {}
    virtual ~LSTMSeqKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
