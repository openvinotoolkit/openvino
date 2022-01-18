// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fully_connected_kernel_base.h"
#include <vector>

namespace kernel_selector {

class FullyConnectedKernelMMAD : public FullyConnectedKernelBase {
public:
    using Parent = FullyConnectedKernelBase;

    FullyConnectedKernelMMAD() : Parent("fully_connected_gpu_MMAD") {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

    struct FullyConnectedTuningData {
        const size_t pack_size = 4;
        size_t sub_group_size = 8;
        size_t slm_div_factor = 1;
        size_t work_group_size = 1;
        size_t feature_blocks_count;
        size_t unroll_factor;
        size_t full_unroll_factor;
    };

protected:
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& kd) const override;
    DispatchData SetDefault(const fully_connected_params& params, int autoTuneIndex = -1) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION,
                 FusedOpType::ELTWISE };
    }
    bool Validate(const Params& params, const optional_params& options) const override;
    FullyConnectedTuningData GetTuningParams(const fully_connected_params& params) const;
};
}  // namespace kernel_selector
