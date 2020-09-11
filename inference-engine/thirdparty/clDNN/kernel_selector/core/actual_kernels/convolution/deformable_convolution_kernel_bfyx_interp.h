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

namespace kernel_selector {

class DeformableConvolutionKernel_bfyx_interp : public common_kernel_base {
public:
    DeformableConvolutionKernel_bfyx_interp() : common_kernel_base("deformable_convolution_gpu_bfyx_interp") {}
    virtual ~DeformableConvolutionKernel_bfyx_interp() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;

protected:
    virtual CommonDispatchData SetDefault(const convolution_params& params) const;
    virtual JitConstants GetJitConstants(const convolution_params& params) const;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
