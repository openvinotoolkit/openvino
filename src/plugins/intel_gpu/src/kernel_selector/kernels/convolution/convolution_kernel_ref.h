// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_Ref : public ConvolutionKernelBase {
public:
    ConvolutionKernel_Ref() : ConvolutionKernelBase("convolution_gpu_ref") {}
    virtual ~ConvolutionKernel_Ref() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params &params) const override {
        if (params.inputs[0].Dimentions() == 4)
            return (params.groups > 1) ? WeightsLayout::goiyx : WeightsLayout::oiyx;
        else
            return (params.groups > 1) ? WeightsLayout::goizyx : WeightsLayout::oizyx;
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

    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;
    bool Validate(const Params& params) const override;
};
}  // namespace kernel_selector
