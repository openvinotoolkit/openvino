// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pooling_kernel_base.h"

namespace kernel_selector {
class PoolingKernelGPURef : public PoolingKernelBase {
public:
    PoolingKernelGPURef() : PoolingKernelBase("pooling_gpu_ref") {}
    virtual ~PoolingKernelGPURef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }

protected:
    JitConstants GetJitConstants(const pooling_params& params, DispatchData dispatchData) const override;
};
}  // namespace kernel_selector
