// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_bfyx_to_bfyx_f16 : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;

    explicit ConvolutionKernel_bfyx_to_bfyx_f16(std::string kernel_name = "convolution_gpu_bfyx_to_bfyx_f16");
    virtual ~ConvolutionKernel_bfyx_to_bfyx_f16() {}

    KernelsData GetKernelsDataForAutoTune(const Params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsData GetTunedKernelsDataByIndex(const Params& params,
                                           int autoTuneIndex = -1) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params&) const override {
        return WeightsLayout::os_is_yx_isv16_osv16;
    }
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }

    bool NeedPaddedInput() const override { return false; }
    bool Validate(const Params& p) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    struct AutoTuneOption {
        size_t blockWidth;
        std::string exeMode;
    };
    AutoTuneOption GetAutoTuneOptions(const Params& arg, int autoTuneIndex) const;

    std::vector<AutoTuneOption> autoTuneOptions;
};
}  // namespace kernel_selector
