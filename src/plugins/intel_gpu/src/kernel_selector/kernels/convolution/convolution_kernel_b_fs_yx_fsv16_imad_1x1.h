// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>
#include <string>

namespace kernel_selector {

class Convolution_kernel_b_fs_yx_fsv16_imad_1x1 : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    Convolution_kernel_b_fs_yx_fsv16_imad_1x1();
    virtual ~Convolution_kernel_b_fs_yx_fsv16_imad_1x1() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsData GetKernelsDataForAutoTune(const Params & params) const override;
    KernelsData GetTunedKernelsDataByIndex(const Params & params, int autoTuneIndex = -1) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;
    bool NeedPaddedInput() const override { return true; }
    WeightsLayout GetPreferredWeightsLayout(const convolution_params&) const override;

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }

    struct AutoTuneParams {
        size_t out_block_spatial;
        size_t out_block_features;
        size_t feature_slm_split;
        std::string exe_mode;
    };
    std::vector<AutoTuneParams> all_tune_params;

    bool ValidateAutoTuneParams(const convolution_params& params, const AutoTuneParams& tune_params) const;
    AutoTuneParams GetAutoTuneParams(const convolution_params& params, int index) const;

    float EstimateOccupancy(const convolution_params& params, const AutoTuneParams& tune) const;
    float EstimateSLMUsage(const convolution_params& params, const AutoTuneParams& tune) const;
};
}  // namespace kernel_selector
