// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pooling_kernel_base.h"
#include <vector>

namespace kernel_selector {
class PoolingKerneGPU_b_fs_yx_fsv4 : public PoolingKernelBase {
public:
    PoolingKerneGPU_b_fs_yx_fsv4() : PoolingKernelBase("pooling_gpu_b_fs_yx_fsv4") {}
    virtual ~PoolingKerneGPU_b_fs_yx_fsv4() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const pooling_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }

protected:
    JitConstants GetJitConstants(const pooling_params& params, DispatchData dispatchData) const override;
};
}  // namespace kernel_selector
