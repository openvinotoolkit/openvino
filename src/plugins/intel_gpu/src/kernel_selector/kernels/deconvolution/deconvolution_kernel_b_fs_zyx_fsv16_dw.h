// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "deconvolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class DeconvolutionKernel_b_fs_zyx_fsv16_dw : public DeconvolutionKernelBase {
public:
    using Parent = DeconvolutionKernelBase;

    DeconvolutionKernel_b_fs_zyx_fsv16_dw() : DeconvolutionKernelBase("deconvolution_gpu_b_fs_zyx_fsv16_dw") {}
    virtual ~DeconvolutionKernel_b_fs_zyx_fsv16_dw() {}
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const deconvolution_params& p) const override {
        if (p.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16)
            return WeightsLayout::gs_oiyx_gsv16;
        else
            return WeightsLayout::gs_oizyx_gsv16;
    }
    bool Validate(const Params& p) const override;
    CommonDispatchData SetDefault(const deconvolution_params& arg) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    JitConstants GetJitConstants(const deconvolution_params& params) const override;

    enum class weights_preload {
        none,
        line,
        all
    };
    enum class input_preload {
        none,
        line
    };

    struct dispatch_params {
        size_t block_size_x;
        input_preload preload_input;
        weights_preload preload_weights;
    };
    dispatch_params GetDispatchParams(const deconvolution_params& params) const;
    float EstimateRegPressure(const deconvolution_params& params, const dispatch_params& disp_params) const;

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE,
            FusedOpType::QUANTIZE
        };
    }
};
}  // namespace kernel_selector
