// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "deconvolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class DeconvolutionKernel_b_fs_zyx_fsv16 : public DeconvolutionKernelBase {
public:
    using Parent = DeconvolutionKernelBase;

    DeconvolutionKernel_b_fs_zyx_fsv16() : DeconvolutionKernelBase("gen9_common_conv_bwd_data") {}
    virtual ~DeconvolutionKernel_b_fs_zyx_fsv16() {}
    ParamsKey GetSupportedKey() const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const deconvolution_params& p) const override {
        if (p.outputs[0].Dimentions() == 4)
            return WeightsLayout::is_os_yx_isv16_osv16;
        else
            return WeightsLayout::is_os_zyx_isv16_osv16;
    }
    bool Validate(const Params& p, const optional_params& o) const override;
    CommonDispatchData SetDefault(const deconvolution_params& arg) const override;
    JitConstants GetJitConstants(const deconvolution_params& params) const override;

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE,
            FusedOpType::SCALE,
            FusedOpType::QUANTIZE
        };
    }
};
}  // namespace kernel_selector
