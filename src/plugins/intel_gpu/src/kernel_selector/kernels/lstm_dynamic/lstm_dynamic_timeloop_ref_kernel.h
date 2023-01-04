// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_dynamic_timeloop_kernel_base.h"

namespace kernel_selector {
class LSTM_DynamicTimeloopKernelRef : public LSTM_DynamicTimeloopKernelBase {
public:
    LSTM_DynamicTimeloopKernelRef() : LSTM_DynamicTimeloopKernelBase("lstm_dynamic_timeloop_ref") {}

    virtual ~LSTM_DynamicTimeloopKernelRef() {}
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;

protected:
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
