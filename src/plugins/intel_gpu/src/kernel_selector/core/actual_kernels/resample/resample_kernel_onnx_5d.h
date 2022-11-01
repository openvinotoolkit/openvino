// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "resample_kernel_base.h"

namespace kernel_selector {
class ResampleKernelOnnx5d : public ResampleKernelBase {
public:
    using Parent = ResampleKernelBase;
    ResampleKernelOnnx5d() : ResampleKernelBase("resample_onnx_5d") {}
    virtual ~ResampleKernelOnnx5d() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants GetJitConstants(const resample_params& params) const override;
    DispatchData SetDefault(const resample_params& arg) const override;
    Datatype GetUnitType(const base_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ELTWISE,
                 FusedOpType::ACTIVATION };
    }
private:
    size_t GetOptimalBlockSize(const resample_params& params) const;
};
}  // namespace kernel_selector
