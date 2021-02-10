/*
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
*/

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
