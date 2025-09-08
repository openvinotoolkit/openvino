// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "convolution_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_bfyx_to_fs_byx_fsv32 : public ConvolutionKernelBase {
public:
    ConvolutionKernel_bfyx_to_fs_byx_fsv32();
    virtual ~ConvolutionKernel_bfyx_to_fs_byx_fsv32() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsData GetKernelsDataForAutoTune(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
    KernelsData GetTunedKernelsDataByIndex(const Params& params,
                                           int autoTuneIndex = -1) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &) const override {
        return WeightsLayout::os_iyx_osv32__ai32;
    }

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }

    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
    bool NeedPaddedInput() const override { return true; }

private:
    struct AutoTuneOption {
        size_t blockWidth;
        size_t blockHeight;
        std::string exeMode;
    };

    std::vector<AutoTuneOption> autoTuneOptions;
    AutoTuneOption GetAutoTuneOptions(const Params& arg, int autoTuneIndex) const;
};

}  // namespace kernel_selector
