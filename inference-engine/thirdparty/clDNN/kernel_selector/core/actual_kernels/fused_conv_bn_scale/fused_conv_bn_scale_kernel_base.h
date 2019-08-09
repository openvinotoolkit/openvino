/*
// Copyright (c) 2018 Intel Corporation
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
*/

#pragma once

#include "weight_bias_kernel_base.h"
#include "actual_kernels/convolution/convolution_params.h"
#include <string>
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// fused_conv_bn_scale_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fused_conv_bn_scale_params : public weight_bias_params {
    fused_conv_bn_scale_params() : weight_bias_params(KernelType::FUSED_CONV_BN_SCALE) {}

    uSize filterSize;
    uSize stride;
    uSize dilation;
    uSize padding;
    uint32_t split = 1;
    bool fused_in_training = false;
    bool scale_bias = false;
    float epsilon = 0.00001f;

    ParamsKey GetParamsKey() const override {
        ParamsKey k = weight_bias_params::GetParamsKey();

        if (split > 1) {
            k.EnableSplitSupport();
        }

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// fused_conv_bn_scale_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fused_conv_bn_scale_optional_params : weight_bias_optional_params {
    fused_conv_bn_scale_optional_params() : weight_bias_optional_params(KernelType::FUSED_CONV_BN_SCALE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// fused_conv_bn_scale_kernel_base
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class fused_conv_bn_scale_kernel_base : public WeightBiasKernelBase {
public:
    using WeightBiasKernelBase::WeightBiasKernelBase;
    virtual ~fused_conv_bn_scale_kernel_base() {}

    using DispatchData = CommonDispatchData;

protected:
    virtual std::vector<WeightsLayout> GetSupportedWeightLayouts(const fused_conv_bn_scale_params&) const = 0;
    virtual std::string GetKernelName(const fused_conv_bn_scale_params&) const { return kernelName; }
    bool Validate(const Params& p, const optional_params& o) const override;
    virtual JitConstants GetJitConstants(const fused_conv_bn_scale_params& params, const DispatchData& kd) const;
    virtual DispatchData SetDefault(const fused_conv_bn_scale_params& params) const;
    static bool CheckWorkGroups(const DispatchData&);
    KernelsData GetCommonKernelsData(const Params& params, const optional_params& options, float estimated_time) const;
};
}  // namespace kernel_selector
