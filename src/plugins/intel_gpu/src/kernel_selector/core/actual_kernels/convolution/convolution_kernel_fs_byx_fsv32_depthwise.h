// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "convolution_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_fs_byx_fsv32_depthwise : public ConvolutionKernelBase {
public:
    ConvolutionKernel_fs_byx_fsv32_depthwise();
    virtual ~ConvolutionKernel_fs_byx_fsv32_depthwise() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsData GetKernelsDataForAutoTune(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    KernelsData GetTunedKernelsDataByIndex(const Params& params,
                                           const optional_params& options,
                                           int autoTuneIndex = -1) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &) const override {
        return WeightsLayout::gs_oiyx_gsv32;
    }

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }

    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
    bool NeedPaddedInput() const override { return true; }

private:
    struct AutoTuneOption {
        size_t blockWidth;
        std::string exeMode;
    };

    std::vector<AutoTuneOption> autoTuneOptions;
    AutoTuneOption GetAutoTuneOptions(const Params& arg, int autoTuneIndex) const;
    size_t getInputWidth(const convolution_params &arg, size_t blockWidth) const;
    size_t getMinRegisterUsage(const convolution_params &arg, size_t blockWidth) const;
};

}  // namespace kernel_selector
