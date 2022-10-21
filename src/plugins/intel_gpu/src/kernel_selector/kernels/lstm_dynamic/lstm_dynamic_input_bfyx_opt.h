// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_dynamic_input_kernel_base.h"

namespace kernel_selector {
class LSTM_DynamicInputKernelBfyxOpt : public LSTM_DynamicInputKernelBase {
public:
    LSTM_DynamicInputKernelBfyxOpt() : LSTM_DynamicInputKernelBase("lstm_dynamic_input_bfyx_opt") {}

    virtual ~LSTM_DynamicInputKernelBfyxOpt() {}
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;

protected:
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& p, const optional_params& o) const override;

private:
    const uint32_t simd_size = 8;
};
}  // namespace kernel_selector
