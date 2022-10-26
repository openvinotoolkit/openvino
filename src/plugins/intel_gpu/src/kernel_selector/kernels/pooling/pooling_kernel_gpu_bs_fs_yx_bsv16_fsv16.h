// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pooling_kernel_base.h"
#include <vector>

namespace kernel_selector {
class Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16 : public PoolingKernelBase {
public:
    Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16() : PoolingKernelBase("pooling_gpu_bs_fs_yx_bsv16_fsv16") {}
    virtual ~Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params&, const optional_params&) const override;
    DispatchData SetDefault(const pooling_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION};
    }

protected:
    JitConstants GetJitConstants(const pooling_params& params, DispatchData dispatchData) const override;
};
}  // namespace kernel_selector
