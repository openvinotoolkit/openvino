// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gemm_kernel_base.h"
#include <vector>

namespace kernel_selector {
class GemmKernelMMADslmInt8 : public GemmKernelBase {
public:
    using Parent = GemmKernelBase;
    using DispatchData = CommonDispatchData;
    struct GemmTuningData {
        size_t size_m;
        size_t size_n;
        size_t size_k;

        const size_t slm_tile_size = 32;
        const size_t simd_size = 8;
        const size_t pack_size = 4;
        const size_t max_slm_preloading_size = 256;
        size_t slm_decimation_factor = 2;
    };

    GemmKernelMMADslmInt8() : GemmKernelBase("gemm_mmad_int8_slm") {}

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
    JitConstants GetJitConstants(const gemm_params& params) const override;
    DispatchData SetDefault(const gemm_params& params) const override;
    GemmTuningData InitGemmTuningData(const gemm_params& params) const;
    GemmTuningData SetTuningParams(const gemm_params& params) const;
    size_t GetMmadOperationsNumber(const GemmTuningData& tuning_data) const;
    bool HasLeftovers(const GemmTuningData& tuning_data) const;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
};
}  // namespace kernel_selector
