// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "fully_connected_kernel_base.h"

namespace kernel_selector {

class FullyConnectedKernelIMAD : public FullyConnectedKernelBase {
public:
    using Parent = FullyConnectedKernelBase;

    FullyConnectedKernelIMAD() : Parent("fully_connected_gpu_imad") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    bool Validate(const Params& params) const override;
    DispatchData SetDefault(const fully_connected_params& params, int autoTuneIndex = -1, int kernel_number = 0) const override;
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION,
                 FusedOpType::ELTWISE };
    }

    struct FullyConnectedTuningData {
        const size_t pack_size = 4;
        size_t sub_group_size = 8;
        size_t tile_ofm = 1;
        size_t tile_batch = 1;
        size_t slm_div_factor = 1;
        size_t work_group_size = 1;
        size_t in_f_blocks_number;
        size_t work_groups_number;
    };

    FullyConnectedTuningData GetTuningParams(const fully_connected_params& params) const;
    float EstimateOccupancy(const fully_connected_params& params, size_t tile_ofm, size_t tile_batch, size_t slm_div_factor = 1) const;
};
}  // namespace kernel_selector
