// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "sdpa_kernel_base.h"

namespace kernel_selector {
class SDPAKernelRef : public SDPAKernelBase {
public:
    using Parent = SDPAKernelBase;
    SDPAKernelRef() : SDPAKernelBase("sdpa_ref") {}
    virtual ~SDPAKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
    CommonDispatchData SetDefault(const sdpa_params& params) const;
    JitConstants GetJitConstants(const sdpa_params& params) const;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {};
    }
};
}  // namespace kernel_selector
