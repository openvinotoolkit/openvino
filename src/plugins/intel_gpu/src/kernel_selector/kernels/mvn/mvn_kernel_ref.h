// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvn_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {
class MVNKernelRef : public MVNKernelBase {
public:
    using Parent = MVNKernelBase;
    MVNKernelRef() : MVNKernelBase("mvn_gpu_ref") {}
    virtual ~MVNKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const mvn_params& params, DispatchData dispatchData) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE,
            FusedOpType::REORDER
        };
    }
    std::string GetKernelName(const mvn_params&) const override;
};
}  // namespace kernel_selector
