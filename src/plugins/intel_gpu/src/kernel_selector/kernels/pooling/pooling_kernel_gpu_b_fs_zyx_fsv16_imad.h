// Copyright (C) 2018-2025 Intel Corporation
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

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
    DispatchData SetDefault(const pooling_params& params) const override;
    bool Validate(const Params&) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }

protected:
    JitConstants GetJitConstants(const pooling_params& params, DispatchData dispatchData) const override;
    bool IsGlobalPooling(const pooling_params& params) const;
};
}  // namespace kernel_selector
