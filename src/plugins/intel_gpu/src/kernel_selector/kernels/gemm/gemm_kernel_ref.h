// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gemm_kernel_base.h"
#include <vector>

namespace kernel_selector {
class GemmKernelRef : public GemmKernelBase {
public:
    using Parent = GemmKernelBase;
    GemmKernelRef() : GemmKernelBase("gemm_ref") {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION,
                 FusedOpType::ELTWISE };
    }
    bool Validate(const Params& params, const optional_params& options) const override;
    JitConstants GetJitConstants(const gemm_params& params) const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params, const optional_params& /*options*/) const override;
};
}  // namespace kernel_selector
