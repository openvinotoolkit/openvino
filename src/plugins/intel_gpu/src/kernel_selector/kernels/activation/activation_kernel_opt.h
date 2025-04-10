// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "activation_kernel_base.h"

#include <vector>

namespace kernel_selector {
class ActivationKernelOpt : public ActivationKernelBase {
public:
    using Parent = ActivationKernelBase;
    ActivationKernelOpt() : Parent("activation_opt") {}
    virtual ~ActivationKernelOpt() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    static const int NUM_COLS_WI = 4;
    DispatchData SetDefault(const activation_params& arg) const override;
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const activation_params& params, DispatchData dispatchData) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {FusedOpType::QUANTIZE,
                FusedOpType::ACTIVATION};
    }
};
}  // namespace kernel_selector
