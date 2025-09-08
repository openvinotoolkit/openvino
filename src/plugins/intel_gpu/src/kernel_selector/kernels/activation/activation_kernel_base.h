// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// activation_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct activation_params : public base_params {
    activation_params() : base_params(KernelType::ACTIVATION) {}

    MultiDataTensor inputActivationParams;

    ParamsKey GetParamsKey() const override {
        auto k = base_params::GetParamsKey();
        if (!inputActivationParams.empty()) {
            k.EnableActivationAdditionalParamsAsInput();
        }
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// activation_fuse_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct activation_fuse_params : fuse_params {
    explicit activation_fuse_params(base_activation_params param)
    : fuse_params(KernelType::ACTIVATION)
    , param(param) {}

    base_activation_params param;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ActivationKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ActivationKernelBase : public KernelBaseOpenCL {
public:
    using DispatchData = CommonDispatchData;
    using KernelBaseOpenCL::KernelBaseOpenCL;

    virtual ~ActivationKernelBase() {}

protected:
    bool Validate(const Params& p) const override;
    virtual JitConstants GetJitConstants(const activation_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const activation_params& arg) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
