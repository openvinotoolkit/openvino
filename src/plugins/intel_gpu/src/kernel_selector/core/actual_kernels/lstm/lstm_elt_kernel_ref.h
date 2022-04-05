// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_elt_kernel_base.h"

namespace kernel_selector {
class LSTMEltKernelRef : public LSTMEltKernelBase {
public:
    LSTMEltKernelRef() : LSTMEltKernelBase("lstm_elt_gpu_bfyx_ref") {}
    virtual ~LSTMEltKernelRef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
