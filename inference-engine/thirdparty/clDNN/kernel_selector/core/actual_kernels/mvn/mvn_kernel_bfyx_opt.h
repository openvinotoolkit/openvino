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


#pragma once

#include "mvn_kernel_base.h"

namespace kernel_selector {
class MVNKernelBfyxOpt : public MVNKernelBase {
public:
    MVNKernelBfyxOpt() : MVNKernelBase("mvn_gpu_bfyx_opt") {}
    virtual ~MVNKernelBfyxOpt() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    using Parent = MVNKernelBase;

private:
    DispatchData SetDefault(const mvn_params& params) const override;
    JitConstants GetJitConstants(const mvn_params& params, MVNKernelBase::DispatchData kd) const override;
};
}  // namespace kernel_selector
