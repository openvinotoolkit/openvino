// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reduce_kernel_base.h"

namespace kernel_selector {

class ReduceKernelWeightedX16 : public ReduceKernelBase {
public:
    ReduceKernelWeightedX16() : ReduceKernelBase("weighted_reduce_x16") {}
    ~ReduceKernelWeightedX16() override = default;

    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& params) const override;
    CommonDispatchData SetDefault(const reduce_params& params) const override;
    JitConstants GetJitConstants(const reduce_params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;

protected:
    bool SupportsWeightedReduce() const override {
        return true;
    }
};

}  // namespace kernel_selector
