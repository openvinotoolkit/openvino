// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "rms_kernel_base.h"

namespace kernel_selector {
class RMSKernelRef : public RMSKernelBase {
public:
    using Parent = RMSKernelBase;
    RMSKernelRef() : RMSKernelBase("rms_gpu_ref") {}
    virtual ~RMSKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE
        };
    }
    JitConstants GetJitConstants(const rms_params& params, DispatchData dispatchData) const override;
};
}  // namespace kernel_selector
