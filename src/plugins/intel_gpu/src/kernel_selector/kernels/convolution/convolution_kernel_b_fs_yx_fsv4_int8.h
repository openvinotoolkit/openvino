// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"
#include <string>

namespace kernel_selector {

class ConvolutionKernel_b_fs_yx_fsv4_int8 : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_b_fs_yx_fsv4_int8() : Parent("convolution_gpu_b_fs_yx_fsv4_int8") {}
    virtual ~ConvolutionKernel_b_fs_yx_fsv4_int8() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params&) const override {
        return WeightsLayout::os_is_yx_osv16_isv4;
    }

    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const override;
    bool Validate(const Params& p) const override;
    bool NeedPaddedInput() const override { return true; }
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }
};
}  // namespace kernel_selector
