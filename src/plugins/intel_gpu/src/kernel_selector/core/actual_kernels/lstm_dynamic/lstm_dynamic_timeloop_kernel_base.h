// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lstm_dynamic_timeloop_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct lstm_dynamic_timeloop_params : public base_params {
    lstm_dynamic_timeloop_params() : base_params(KernelType::LSTM_DYNAMIC_TIMELOOP) {}

    DataTensor recurrent;
    DataTensor hidden;
    DataTensor cell;
    DataTensor last_hidden_output;
    DataTensor last_cell_output;

    float clip = 0.0f;
    bool input_forget = false;
    bool has_hidden = false;
    bool has_cell = false;
    bool has_last_hidden_output = false;
    bool has_last_cell_output = false;
    int32_t direction = 1;

    void set_hidden(const DataTensor& v) {
        hidden = v;
        has_hidden = true;
    }

    void set_cell(const DataTensor& v) {
        cell = v;
        has_cell = true;
    }

    void set_last_hidden_output(const DataTensor& v) {
        last_hidden_output = v;
        has_last_hidden_output = true;
    }

    void set_last_cell_output(const DataTensor& v) {
        last_cell_output = v;
        has_last_cell_output = true;
    }

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();

        if (has_hidden) {
            k.EnableLSTMGEMMHidden();
        }

        if (has_cell) {
            k.EnableLSTMEltCell();
        }

        if (has_last_hidden_output) {
            k.EnableLSTMDyanmicOptionalHiddenOutput();
        }

        if (has_last_cell_output) {
            k.EnableLSTMDyanmicOptionalCellOutput();
        }

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lstm_dynamic_timeloop_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct lstm_dynamic_optional_params : optional_params {
    lstm_dynamic_optional_params() : optional_params(KernelType::LSTM_DYNAMIC_TIMELOOP) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// LSTM_DynamicTimeloopKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class LSTM_DynamicTimeloopKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~LSTM_DynamicTimeloopKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    virtual JitConstants GetJitConstants(const lstm_dynamic_timeloop_params& params) const;
    static DispatchData SetDefault(const lstm_dynamic_timeloop_params& params);
    KernelsData GetCommonKernelsData(const Params& params,
                                     const optional_params& optParams) const;
    void SetKernelArguments(const lstm_dynamic_timeloop_params& params, clKernelData& k_data) const;
    bool Validate(const Params& p, const optional_params&) const override {
        if (p.GetType() != KernelType::LSTM_DYNAMIC_TIMELOOP) {
            return false;
        }

        return true;
    }
};
}  // namespace kernel_selector
