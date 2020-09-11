/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once

#include "lstm_dynamic_input_kernel_base.h"

namespace kernel_selector {
class LSTM_DynamicInputKernelBfyxOpt : public LSTM_DynamicInputKernelBase {
public:
    LSTM_DynamicInputKernelBfyxOpt() : LSTM_DynamicInputKernelBase("lstm_dynamic_input_bfyx_opt") {}

    virtual ~LSTM_DynamicInputKernelBfyxOpt() {}
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;

protected:
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& p, const optional_params& o) const override;

private:
    const uint32_t simd_size = 8;
};
}  // namespace kernel_selector
