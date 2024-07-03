// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device_info.hpp"
#include "convolution_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_bfyx_os_iyx_osv32 : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_bfyx_os_iyx_osv32();
    virtual ~ConvolutionKernel_bfyx_os_iyx_osv32() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsData GetKernelsDataForAutoTune(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &) const override {
        return WeightsLayout::os_iyx_osv32;
    }

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }

    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    bool Validate(const Params& p) const override;
    bool NeedPaddedInput() const override { return true; }
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;

private:
    struct AutoTuneOption {
        size_t blockWidth;
        size_t blockHeight;
        std::string exeMode;
    };

    AutoTuneOption GetAutoTuneOptions(const Params& arg, int autoTuneIndex) const;

    std::vector<AutoTuneOption> autoTuneOptions = {};
};
}  // namespace kernel_selector
