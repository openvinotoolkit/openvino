// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>
#include <string>

namespace kernel_selector {

class ConvolutionKernel_imad_b_fs_yx_fsv4_1x1 : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_imad_b_fs_yx_fsv4_1x1();

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;

    WeightsLayout GetPreferredWeightsLayout(const convolution_params &) const override {
        return WeightsLayout::os_is_yx_osv16_isv4;
    }

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }

    struct AutoTuneParams {
        size_t simd;
        size_t features_per_wi;
        size_t lwg_depth;
        bool force_prefetch;
        std::string exeMode;
    };

    std::vector<AutoTuneParams> all_tune_params;

    bool ValidateAutoTuneParams(const convolution_params& params, const AutoTuneParams& tune_params) const;
    AutoTuneParams GetAutoTuneParams(const convolution_params& params, int index) const;
    KernelsData GetTunedKernelsDataByIndex(const Params& params,
                                           int autoTuneIndex = -1) const override;
    KernelsData GetKernelsDataForAutoTune(const Params& params) const override;
};
}  // namespace kernel_selector
