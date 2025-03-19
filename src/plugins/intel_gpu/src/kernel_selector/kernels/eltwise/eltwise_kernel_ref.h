// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "eltwise_kernel_base.h"

namespace kernel_selector {
class EltwiseKernelRef : public EltwiseKernelBase {
public:
    EltwiseKernelRef() : EltwiseKernelBase("generic_eltwise_ref") {}
    virtual ~EltwiseKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::QUANTIZE,
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE,
            FusedOpType::REORDER
        };
    }

    JitConstants GetJitConstants(const eltwise_params& params) const override;

protected:
    bool Validate(const Params& p) const override;
};
}  // namespace kernel_selector
