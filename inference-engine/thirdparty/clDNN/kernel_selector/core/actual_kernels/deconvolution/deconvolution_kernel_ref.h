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

#include "deconvolution_kernel_base.h"

namespace kernel_selector {

class DeconvolutionKernelRef : public DeconvolutionKernelBase {
public:
    DeconvolutionKernelRef() : DeconvolutionKernelBase("deconvolution_gpu_ref") {}
    virtual ~DeconvolutionKernelRef() {}

    ParamsKey GetSupportedKey() const override;

protected:
    CommonDispatchData SetDefault(const deconvolution_params& params) const override;
    JitConstants GetJitConstants(const deconvolution_params& params) const override;
};
}  // namespace kernel_selector