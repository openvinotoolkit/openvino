// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "depth_to_space_kernel_base.h"

namespace kernel_selector {
class DepthToSpaceKernelBlock2Opt : public DepthToSpaceKernelBase {
public:
    using Parent = DepthToSpaceKernelBase;

    DepthToSpaceKernelBlock2Opt() : DepthToSpaceKernelBase("depth_to_space_block2_opt") {}
    virtual ~DepthToSpaceKernelBlock2Opt() {}

    bool Validate(const Params&) const override;
    JitConstants GetJitConstants(const depth_to_space_params& params) const override;
    CommonDispatchData SetDefault(const depth_to_space_params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
