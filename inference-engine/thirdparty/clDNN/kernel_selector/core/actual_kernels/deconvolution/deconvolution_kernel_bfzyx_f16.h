//
// Copyright (c) 2019 Intel Corporation
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

class DeconvolutionKernel_bfzyx_f16 : public DeconvolutionKernelBase {
public:
    using Parent = DeconvolutionKernelBase;

    DeconvolutionKernel_bfzyx_f16() : DeconvolutionKernelBase("gen9_common_conv_bwd_data") {}
    virtual ~DeconvolutionKernel_bfzyx_f16() {}
    ParamsKey GetSupportedKey() const override;

protected:
    std::vector<WeightsLayout> GetSupportedWeightLayouts(const deconvolution_params&) const override {
        return {
            WeightsLayout::i_o_zyx_o16_i16,
        };
    }
    bool Validate(const Params& p, const optional_params& o) const override;
    CommonDispatchData SetDefault(const deconvolution_params& arg) const override;
    JitConstants GetJitConstants(const deconvolution_params& params) const override;
};
}  // namespace kernel_selector
