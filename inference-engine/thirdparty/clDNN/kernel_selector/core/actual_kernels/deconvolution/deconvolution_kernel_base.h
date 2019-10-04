// Copyright (c) 2016 Intel Corporation
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


#pragma once

#include "weight_bias_kernel_base.h"
#include "kernel_selector_params.h"
#include <string>
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// deconvolution_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct deconvolution_params : public weight_bias_params {
    deconvolution_params() : weight_bias_params(KernelType::DECONVOLUTION) {}

    uSize filterSize;
    uSize stride;
    uSize dilation;
    uSize padding;
    uint32_t split = 1;
    uint32_t groups = 1;
    bool depthwise_separable_opt = false;
    bool fused_eltwise = false;

    std::string to_string() const override;

    ParamsKey GetParamsKey() const override {
        ParamsKey k = weight_bias_params::GetParamsKey();

        if (split > 1) {
            k.EnableSplitSupport();
        }

        if (dilation.x != 1 || dilation.y != 1 || dilation.z != 1) {
            k.EnableDilation();
        }

        if (depthwise_separable_opt) {
            k.EnableDepthwiseSeparableOpt();
        }

        if (groups > 1 && !depthwise_separable_opt) {
            k.EnableGroupedConvolution();
        }

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// deconvolution_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct deconvolution_optional_params : weight_bias_optional_params {
    deconvolution_optional_params() : weight_bias_optional_params(KernelType::DECONVOLUTION) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DeconvolutionKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DeconvolutionKernelBase : public WeightBiasKernelBase {
public:
    using WeightBiasKernelBase::WeightBiasKernelBase;
    virtual ~DeconvolutionKernelBase() {}

    using DispatchData = CommonDispatchData;

protected:
    virtual KernelsData GetKernelsData(const Params& params, const optional_params& options) const;
    virtual JitConstants GetJitConstants(const deconvolution_params& params) const;
    virtual DispatchData SetDefault(const deconvolution_params& params) const;
    virtual std::vector<WeightsLayout> GetSupportedWeightLayouts(const deconvolution_params&) const {
        return {
            WeightsLayout::oiyx,
            WeightsLayout::iyxo,
            WeightsLayout::yxio,
            WeightsLayout::oyxi,
            WeightsLayout::oizyx
        };
    }
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector