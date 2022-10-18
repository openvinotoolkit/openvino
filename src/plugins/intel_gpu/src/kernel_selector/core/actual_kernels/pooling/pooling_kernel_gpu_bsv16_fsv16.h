// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pooling_kernel_base.h"
#include <vector>

namespace kernel_selector {
class PoolingKernel_bsv16_fsv16 : public PoolingKernelBase {
public:
    using Parent = PoolingKernelBase;

    PoolingKernel_bsv16_fsv16() : PoolingKernelBase("pooling_gpu_bsv16_fsv16") {}

    virtual ~PoolingKernel_bsv16_fsv16() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants GetJitConstants(const pooling_params& params, DispatchData dispatchData) const override;
    DispatchData SetDefault(const pooling_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }
};
}  // namespace kernel_selector
