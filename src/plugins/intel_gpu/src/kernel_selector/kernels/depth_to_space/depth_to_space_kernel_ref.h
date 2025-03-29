// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "depth_to_space_kernel_base.h"

namespace kernel_selector {
class DepthToSpaceKernelRef : public DepthToSpaceKernelBase {
public:
    using Parent = DepthToSpaceKernelBase;

    DepthToSpaceKernelRef() : DepthToSpaceKernelBase("depth_to_space_ref") {}
    virtual ~DepthToSpaceKernelRef() {}

    CommonDispatchData SetDefault(const depth_to_space_params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const depth_to_space_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::REORDER,
                 FusedOpType::ACTIVATION };
    }
};
}  // namespace kernel_selector
