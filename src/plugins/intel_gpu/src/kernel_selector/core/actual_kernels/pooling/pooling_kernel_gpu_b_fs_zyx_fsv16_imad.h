// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pooling_kernel_base.h"
#include <vector>

namespace kernel_selector {
class PoolingKernelGPU_b_fs_zyx_fsv16_imad: public PoolingKernelBase{
public:
    PoolingKernelGPU_b_fs_zyx_fsv16_imad() : PoolingKernelBase("pooling_gpu_b_fs_zyx_fsv16_imad") {}
    virtual ~PoolingKernelGPU_b_fs_zyx_fsv16_imad() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const pooling_params& params) const override;
    bool Validate(const Params&, const optional_params&) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }

protected:
    JitConstants GetJitConstants(const pooling_params& params, DispatchData dispatchData) const override;
    bool IsGlobalPooling(const pooling_params& params) const;
};
}  // namespace kernel_selector
