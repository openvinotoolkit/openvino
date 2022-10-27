// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lstm_gemm_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct lstm_gemm_params : public base_params {
    lstm_gemm_params() : base_params(KernelType::LSTM_GEMM) {}

    DataTensor weights;
    DataTensor recurrent;
    DataTensor bias;
    DataTensor hidden;
    bool hasBias = false;
    bool hasHidden = false;
    uint32_t direction = 0;
    uint32_t input_direction = 0;  // for bidirectional node fusion in stacked LSTMs
    uint32_t hidden_direction = 0;

    void SetBias(const DataTensor& v) {
        bias = v;
        hasBias = true;
    }

    void SetHidden(const DataTensor& v) {
        hidden = v;
        hasHidden = true;
    }

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();

        if (hasBias) {
            k.EnableLSTMGEMMBias();
        }

        if (hasHidden) {
            k.EnableLSTMGEMMHidden();
        }

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lstm_gemm_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct lstm_gemm_optional_params : optional_params {
    lstm_gemm_optional_params() : optional_params(KernelType::LSTM_GEMM) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// LSTMGemmKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class LSTMGemmKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~LSTMGemmKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    virtual JitConstants GetJitConstants(const lstm_gemm_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params& optParams) const;

    bool Validate(const Params& p, const optional_params&) const override {
        if (p.GetType() != KernelType::LSTM_GEMM) {
            return false;
        }

        return true;
    }
};
}  // namespace kernel_selector
