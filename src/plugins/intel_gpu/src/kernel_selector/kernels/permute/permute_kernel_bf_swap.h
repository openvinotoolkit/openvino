// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "permute_kernel_base.h"

namespace kernel_selector {

class PermuteKernel_bf_swap : public PermuteKernelBase {
public:
    using Parent = PermuteKernelBase;
    using Parent::Parent;
    PermuteKernel_bf_swap() : PermuteKernelBase("permute_bf_swap") {}
    ~PermuteKernel_bf_swap() override = default;

    bool Validate(const Params& p) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const permute_params& params, const CommonDispatchData& dispatch_data) const override;
    CommonDispatchData SetDefault(const permute_params& params) const override;
};

}  // namespace kernel_selector
