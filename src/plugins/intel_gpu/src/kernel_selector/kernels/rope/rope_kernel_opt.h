// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "rope_kernel_base.h"

namespace kernel_selector {
class RoPEKernelOpt : public RoPEKernelBase {
public:
    using Parent = RoPEKernelBase;
    RoPEKernelOpt() : RoPEKernelBase("rope_opt") {}
    virtual ~RoPEKernelOpt() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
protected:
    JitConstants GetJitConstants(const rope_params& params, DispatchData dispatchData) const override;
    DispatchData SetDefault(const rope_params& params) const override;
private:
    size_t GetVecSize(const rope_params& params) const;
};
}  // namespace kernel_selector