// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_dynamic_input_kernel_base.h"

namespace kernel_selector {
class LSTM_DynamicInputKernelRef : public LSTM_DynamicInputKernelBase {
public:
    LSTM_DynamicInputKernelRef() : LSTM_DynamicInputKernelBase("lstm_dynamic_input_ref") {}

    virtual ~LSTM_DynamicInputKernelRef() {}
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;

protected:
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
