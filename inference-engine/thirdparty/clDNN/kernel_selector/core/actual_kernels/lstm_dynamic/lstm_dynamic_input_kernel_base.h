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
#include "weight_bias_params.h"
#include "common_kernel_base.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lstm_dynamic_input_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct lstm_dynamic_input_params : public weight_bias_params {
    lstm_dynamic_input_params() : weight_bias_params(KernelType::LSTM_DYNAMIC_INPUT) {}

    int32_t direction = 1;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lstm_dynamic_input_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct lstm_dynamic_input_optional_params : weight_bias_optional_params {
    lstm_dynamic_input_optional_params() : weight_bias_optional_params(KernelType::LSTM_DYNAMIC_INPUT) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// LSTM_DynamicInputKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class LSTM_DynamicInputKernelBase : public common_kernel_base {
public:
    using common_kernel_base::common_kernel_base;
    virtual ~LSTM_DynamicInputKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    virtual JitConstants GetJitConstants(const lstm_dynamic_input_params& params) const;
    static DispatchData SetDefault(const lstm_dynamic_input_params& params);
    KernelsData GetCommonKernelsData(const Params& params,
                                     const optional_params& optParams,
                                     float estimated_time) const;
    void SetKernelArguments(const lstm_dynamic_input_params& params, clKernelData& k_data) const;

    bool Validate(const Params& p, const optional_params&) const override {
        if (p.GetType() != KernelType::LSTM_DYNAMIC_INPUT) {
            return false;
        }

        return true;
    }
};
}  // namespace kernel_selector
