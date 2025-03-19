// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "resample_kernel_base.h"

namespace kernel_selector {
class ResampleKernelOnnx : public ResampleKernelBase {
public:
    using Parent = ResampleKernelBase;
    ResampleKernelOnnx() : ResampleKernelBase("resample_onnx") {}
    virtual ~ResampleKernelOnnx() = default;

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const resample_params& params) const override;
    DispatchData SetDefault(const resample_params& arg) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {FusedOpType::QUANTIZE, FusedOpType::ELTWISE, FusedOpType::ACTIVATION};
    }

private:
    size_t GetOptimalBlockSize(const resample_params& params) const;
};
}  // namespace kernel_selector
