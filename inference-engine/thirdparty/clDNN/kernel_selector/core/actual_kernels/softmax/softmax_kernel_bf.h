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

#include "softmax_kernel_base.h"

namespace kernel_selector {
class SoftmaxKernel_bf : public SoftmaxKernelBaseBF {
public:
    using Parent = SoftmaxKernelBaseBF;
    SoftmaxKernel_bf() : Parent("softmax_gpu_bf") {}
    virtual ~SoftmaxKernel_bf() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const softmax_params& params, const optional_params& optParams) const override;
};
}  // namespace kernel_selector