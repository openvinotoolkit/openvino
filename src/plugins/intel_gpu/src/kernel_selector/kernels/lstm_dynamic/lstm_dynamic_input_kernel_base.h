// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "weight_bias_params.h"
#include "kernel_base_opencl.h"
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
class LSTM_DynamicInputKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~LSTM_DynamicInputKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    virtual JitConstants GetJitConstants(const lstm_dynamic_input_params& params) const;
    static DispatchData SetDefault(const lstm_dynamic_input_params& params);
    KernelsData GetCommonKernelsData(const Params& params,
                                     const optional_params& optParams) const;
    void SetKernelArguments(const lstm_dynamic_input_params& params, clKernelData& k_data) const;

    bool Validate(const Params& p, const optional_params&) const override {
        if (p.GetType() != KernelType::LSTM_DYNAMIC_INPUT) {
            return false;
        }

        return true;
    }
};
}  // namespace kernel_selector
