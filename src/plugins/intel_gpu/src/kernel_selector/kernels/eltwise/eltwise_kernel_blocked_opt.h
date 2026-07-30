// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "eltwise_kernel_base.h"

namespace kernel_selector {
class EltwiseKernel_blocked_opt : public EltwiseKernelBase {
public:
    EltwiseKernel_blocked_opt() : EltwiseKernelBase("eltwise_blocked_opt") {}
    ~EltwiseKernel_blocked_opt() override {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::QUANTIZE,
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE
        };
    }

protected:
    bool Validate(const Params& p) const override;
    JitConstants MakeLoadJitConstants(const eltwise_params& params, bool useVload8) const override;
    JitConstants GetJitConstants(const eltwise_params& params) const override;
    DispatchData SetDefault(const eltwise_params& params) const override;
    void PrintWorkSize(const DispatchData& dis);
};
}  // namespace kernel_selector
