// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvn_kernel_base.h"

#include <vector>

namespace kernel_selector {
class MVNKernelBfyxOpt : public MVNKernelBase {
public:
    MVNKernelBfyxOpt() : MVNKernelBase("mvn_gpu_bfyx_opt") {}
    virtual ~MVNKernelBfyxOpt() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    using Parent = MVNKernelBase;

private:
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE,
            FusedOpType::REORDER
        };
    }
    DispatchData SetDefault(const mvn_params& params) const override;
    JitConstants GetJitConstants(const mvn_params& params, MVNKernelBase::DispatchData dispatchData) const override;
};
}  // namespace kernel_selector
