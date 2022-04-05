// Copyright (C) 2018-2022 Intel Corporation
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

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    static const int NUM_COLS_WI = 4;
    DispatchData SetDefault(const activation_params& arg) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants GetJitConstants(const activation_params& params, DispatchData dispatchData) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {FusedOpType::QUANTIZE,
                FusedOpType::SCALE,
                FusedOpType::ACTIVATION};
    }
};
}  // namespace kernel_selector
