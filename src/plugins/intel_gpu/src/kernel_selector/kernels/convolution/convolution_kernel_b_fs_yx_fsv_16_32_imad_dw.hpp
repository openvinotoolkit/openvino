// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>
#include <string>

namespace kernel_selector {
class ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw();
    virtual ~ConvolutionKernel_b_fs_yx_fsv_16_32_imad_dw() {}

    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsData GetKernelsDataForAutoTune(const Params& params) const override;
    KernelsData GetTunedKernelsDataByIndex(const Params& params, int autoTuneIndex = -1) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;

protected:
    bool Validate(const Params& params) const override;
    WeightsLayout GetPreferredWeightsLayout(const convolution_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }

    bool NeedPaddedInput() const override { return false; }
    bool HasPaddedInput(const convolution_params& params) const;
    bool ParamsHavePadding(const convolution_params& params) const;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;

    struct AutoTuneParams {
        size_t simd;
        size_t tile_x;
        size_t lws0;
        size_t lws1;
        bool preload_input_slm;
        std::string exeMode;
    };
    std::vector<AutoTuneParams> all_tune_params;

    AutoTuneParams GetAutoTuneParams(const convolution_params& params, int index) const;
    bool ValidateAutoTuneParams(const convolution_params& params, const AutoTuneParams& tparams) const;
};
}  // namespace kernel_selector
