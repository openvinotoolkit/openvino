// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "sdpa_kernel_base.h"

namespace kernel_selector {
class SDPAKernelOpt : public SDPAKernelBase {
public:
    using Parent = SDPAKernelBase;
    SDPAKernelOpt() : SDPAKernelBase("sdpa_opt") {}
    virtual ~SDPAKernelOpt() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
    CommonDispatchData SetDefault(const sdpa_params& params, size_t kernel_idx) const;
    JitConstants GetJitConstants(const sdpa_params& params, size_t kernel_idx) const;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {};
    }
};
}  // namespace kernel_selector
