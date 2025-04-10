// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lrn_kernel_base.h"
#include "vector"

namespace kernel_selector {
class LRNKernelWithinChannel : public LRNKernelBase {
public:
    using Parent = LRNKernelBase;
    LRNKernelWithinChannel() : LRNKernelBase("lrn_gpu_within_channel") {}
    virtual ~LRNKernelWithinChannel() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

private:
    DispatchData SetDefault(const lrn_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ELTWISE,
                 FusedOpType::ACTIVATION };
    }
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const lrn_params& params, const DispatchData& dispatchData) const override;
};
}  // namespace kernel_selector
