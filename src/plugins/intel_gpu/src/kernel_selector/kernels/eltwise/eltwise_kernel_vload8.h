// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "eltwise_kernel_base.h"

namespace kernel_selector {
class EltwiseKernel_vload8 : public EltwiseKernelBase {
public:
    EltwiseKernel_vload8() : EltwiseKernelBase("eltwise_simple_vload8") {}
    virtual ~EltwiseKernel_vload8() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION
        };
    }

protected:
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const eltwise_params& params) const override;
};
}  // namespace kernel_selector
