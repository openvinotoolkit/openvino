// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GroupNormalizationParams
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct multinomial_params : public base_params {
    multinomial_params() : base_params(KernelType::MULTINOMIAL) {}

    ov::element::Type_t output_data_type {};
    bool with_replacement {};
    bool log_probs {};

    ParamsKey GetParamsKey() const override {
        return base_params::GetParamsKey();
    }
};

class MultinomialKernelRef : public KernelBaseOpenCL {
public:
    MultinomialKernelRef() : KernelBaseOpenCL{"multinomial_ref"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

private:
    JitConstants GetJitConstants(const multinomial_params &params) const;
    CommonDispatchData SetDefault(const multinomial_params &params) const;
    bool Validate(const Params &p) const override;
};

} // namespace kernel_selector
