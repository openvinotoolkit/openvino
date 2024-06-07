// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gemm_kernel_base.h"
#include <vector>

namespace kernel_selector {
class GemmKernelTiledOpt : public GemmKernelBase {
public:
    using Parent = GemmKernelBase;

    struct GemmTuningData {
        size_t simd_size = 8;
        size_t tile_m_size = 1;
        size_t tile_k_size = 1;
        size_t tile_n_size = 8;
    };

    GemmKernelTiledOpt() : GemmKernelBase("gemm_tiled_opt") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION,
                 FusedOpType::ELTWISE };
    }
    bool Validate(const Params& params) const override;
    DispatchData SetDefault(const gemm_params& params) const override;
    JitConstants GetJitConstants(const gemm_params& params) const override;
    GemmTuningData SetTuningParams(const gemm_params& params) const;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
