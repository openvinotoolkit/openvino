// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class DeformableConvolutionKernel_bfyx_conv : public ConvolutionKernelBase {
public:
    DeformableConvolutionKernel_bfyx_conv() : ConvolutionKernelBase("deformable_convolution_gpu_bfyx_conv") {}
    virtual ~DeformableConvolutionKernel_bfyx_conv() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;

protected:
    DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    ParamsKey GetSupportedKey() const override;
    WeightsLayout GetPreferredWeightsLayout(const convolution_params&) const override {
        return WeightsLayout::os_is_yx_isv16_osv16;
    }
};
}  // namespace kernel_selector
