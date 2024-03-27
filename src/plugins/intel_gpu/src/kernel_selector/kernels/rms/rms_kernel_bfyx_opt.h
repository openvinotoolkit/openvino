// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "rms_kernel_base.h"

namespace kernel_selector {
class RMSKernelBfyxOpt : public RMSKernelBase {
public:
    using Parent = RMSKernelBase;
    RMSKernelBfyxOpt() : RMSKernelBase("rms_gpu_bfyx_opt") {}
    virtual ~RMSKernelBfyxOpt() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params&) const override;
    DispatchData SetDefault(const rms_params& params) const override;
    JitConstants GetJitConstants(const rms_params& params, DispatchData dispatchData) const override;
    const size_t vec_size = 8;
};
}  // namespace kernel_selector
