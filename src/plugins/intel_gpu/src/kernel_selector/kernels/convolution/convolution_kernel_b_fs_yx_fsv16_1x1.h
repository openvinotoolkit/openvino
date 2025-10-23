// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>
#include <string>

namespace kernel_selector {

class ConvolutionKernel_b_fs_yx_fsv16_1x1 : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;

    ConvolutionKernel_b_fs_yx_fsv16_1x1();
    virtual ~ConvolutionKernel_b_fs_yx_fsv16_1x1() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &) const override {
        return WeightsLayout::os_is_yx_isv16_osv16;
    }
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }
    bool Validate(const Params& p) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;

    struct AutoTuneOption {
        size_t blockWidth;
        std::string exeMode;
    };

    struct ConvolutionTuningData {
        const size_t sub_group_size = 16;
        const size_t feature_block_size = 16;
        size_t slm_div_factor = 1;
        size_t work_group_size = 1;
    };

    std::vector<AutoTuneOption> autoTuneOptions;
    AutoTuneOption GetAutoTuneOptions(const convolution_params& arg, int autoTuneIndex) const;
    ConvolutionTuningData GetTuningParams(const convolution_params& params) const;
    float EstimateOccupancy(const convolution_params& params, const ConvolutionTuningData& tuning_data) const;
};
}  // namespace kernel_selector
