// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_b_fs_yx_fsv16 : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;

    ConvolutionKernel_b_fs_yx_fsv16();
    virtual ~ConvolutionKernel_b_fs_yx_fsv16() {}

    KernelsData GetKernelsDataForAutoTune(const Params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    KernelsData GetTunedKernelsDataByIndex(const Params& params,
                                           int autoTuneIndex = -1) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &p) const override {
        return (p.groups > 1) ? WeightsLayout::g_os_is_yx_isv16_osv16 : WeightsLayout::os_is_yx_isv16_osv16;
    }
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        // FusedOpType::REORDER should be registered explicitly here
        // only when fused_primitive_desc for reorder is added by optimization passes (e.g., remove_redundant_reorder) for corresponding primitive.
        // The typical usage for fused_primitive_desc for convolution is to get original output layout from jitter,
        // so that it can decide whether to fuse eltwise along with reorder.
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION,
                 FusedOpType::REORDER };
    }

    bool NeedPaddedInput() const override { return false; }
    bool Validate(const Params& p) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;

private:
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
    AutoTuneOption GetAutoTuneOptions(const Params& arg, int autoTuneIndex) const;
    ConvolutionTuningData GetTuningParams(const convolution_params& params) const;
    float EstimateOccupancy(const convolution_params& params, const ConvolutionTuningData& tuning_data) const;
};
}  // namespace kernel_selector
