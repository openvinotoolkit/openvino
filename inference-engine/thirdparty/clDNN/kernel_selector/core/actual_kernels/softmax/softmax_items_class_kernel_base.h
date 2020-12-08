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
#include <vector>

namespace kernel_selector {
class SoftmaxItemsClassKernelBase : public SoftmaxKernelBase {
public:
    using SoftmaxKernelBase::SoftmaxKernelBase;
    virtual ~SoftmaxItemsClassKernelBase() {}

protected:
    JitConstants GetJitConstants(const softmax_params& params, DispatchData dispatchData) const override;
    static ParamsKey GetDefaultSupportedKey();
    static std::vector<size_t> GetSoftmaxDimGlobalSizes(SoftmaxDim dim, const DataTensor& output);
};
}  // namespace kernel_selector
