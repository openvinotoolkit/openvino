// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class DeformableConvolutionKernel_bfyx_opt : public ConvolutionKernelBase {
public:
    DeformableConvolutionKernel_bfyx_opt() : ConvolutionKernelBase("deformable_convolution_gpu_bfyx_opt") {}
    virtual ~DeformableConvolutionKernel_bfyx_opt() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    void SetKernelArguments(const convolution_params& params, clKernelData& kernel, size_t idx) const;
    WeightsLayout GetPreferredWeightsLayout(const convolution_params&) const override {
        return WeightsLayout::os_is_yx_isv16_osv16;
    }
};
}  // namespace kernel_selector
