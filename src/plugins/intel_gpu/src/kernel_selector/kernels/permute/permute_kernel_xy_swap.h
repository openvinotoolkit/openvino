// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "permute_kernel_base.h"

namespace kernel_selector {

// Optimized kernel for 4D Transpose that swaps only the last two axes (Y and X).
// Pattern (cldnn order): {0, 1, 3, 2}
// Equivalent IE order:   {0, 1, 3, 2}  (e.g., bfyx -> bfxy, swap last two dims)
//
// Uses SLM-tiled cooperative loads/stores to recover coalesced global memory
// access on both read (input X innermost) and write (output X = input Y).
class PermuteKernel_xy_swap : public PermuteKernelBase {
public:
    using Parent = PermuteKernelBase;
    using Parent::Parent;
    PermuteKernel_xy_swap() : PermuteKernelBase("permute_xy_swap") {}
    ~PermuteKernel_xy_swap() override = default;

    bool Validate(const Params& p) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const override;
    CommonDispatchData SetDefault(const permute_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {FusedOpType::REORDER, FusedOpType::ACTIVATION, FusedOpType::QUANTIZE, FusedOpType::ELTWISE};
    }
};

}  // namespace kernel_selector
