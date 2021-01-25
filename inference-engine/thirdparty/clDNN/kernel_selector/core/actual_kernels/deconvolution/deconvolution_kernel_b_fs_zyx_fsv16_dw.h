//
// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

protected:
    WeightsLayout GetPreferredWeightsLayout(const deconvolution_params& p) const override {
        if (p.output.GetLayout() == DataLayout::b_fs_yx_fsv16)
            return WeightsLayout::gs_oiyx_gsv16;
        else
            return WeightsLayout::gs_oizyx_gsv16;
    }
    bool Validate(const Params& p, const optional_params& o) const override;
    CommonDispatchData SetDefault(const deconvolution_params& arg) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
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
            FusedOpType::SCALE,
            FusedOpType::QUANTIZE
        };
    }
};
}  // namespace kernel_selector
