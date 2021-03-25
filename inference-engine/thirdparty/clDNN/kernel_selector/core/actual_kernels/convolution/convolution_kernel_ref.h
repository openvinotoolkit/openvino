// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_Ref : public ConvolutionKernelBase {
public:
    // The kernel is shared with fused_conv_eltwise_gpu_ref primitive.
    ConvolutionKernel_Ref() : ConvolutionKernelBase("fused_conv_eltwise_gpu_ref") {}
    virtual ~ConvolutionKernel_Ref() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &params) const override {
        if (params.inputs[0].Dimentions() == 4)
            return (params.groups > 1) ? WeightsLayout::goiyx : WeightsLayout::oiyx;
        else
            return (params.groups > 1) ? WeightsLayout::goizyx : WeightsLayout::oizyx;
    }
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }

    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;
    bool Validate(const Params& params, const optional_params& options) const override;
};
}  // namespace kernel_selector
