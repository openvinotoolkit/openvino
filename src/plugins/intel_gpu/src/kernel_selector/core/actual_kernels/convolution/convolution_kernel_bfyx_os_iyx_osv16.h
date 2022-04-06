// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device_info.hpp"
#include "convolution_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_bfyx_os_iyx_osv16 : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_bfyx_os_iyx_osv16();
    virtual ~ConvolutionKernel_bfyx_os_iyx_osv16() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsData GetKernelsDataForAutoTune(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }

    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
    bool NeedPaddedInput() const override { return true; }
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
    size_t GetSubGroupSize(const convolution_params& params) const {
        if (params.engineInfo.computeUnitsCount <= 24) {
            // Smaller # EU tends to be computation bounds.
            // In such case, using larger worksize will result in larger computational inefficiency
            // w.r.t the unalined output feature
            if (params.outputs[0].Feature().v > 8 || !IsSIMDSizeSupported(params.engineInfo, 8)
               || ((params.outputs[0].GetDType() == Datatype::F16) && params.outputs[0].Feature().v == 8)) {
                return 16;
            } else {
                return 8;
            }
        } else {
            return 16;
        }
    }

private:
    struct AutoTuneOption {
        size_t blockWidth;
        size_t blockHeight;
        size_t prefetch;
        std::string exeMode;
    };

    AutoTuneOption GetAutoTuneOptions(const Params& arg, int autoTuneIndex) const;

    std::vector<AutoTuneOption> autoTuneOptions = {};
};
}  // namespace kernel_selector
