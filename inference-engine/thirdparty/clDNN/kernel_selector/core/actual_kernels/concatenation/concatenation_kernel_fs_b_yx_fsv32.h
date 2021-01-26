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

#pragma once

#include "concatenation_kernel_base.h"

namespace kernel_selector {

class ConcatenationKernel_fs_b_yx_fsv32 : public ConcatenationKernelBase {
public:
    ConcatenationKernel_fs_b_yx_fsv32() : ConcatenationKernelBase("concatenation_gpu_fs_b_yx_fsv32") {}
    virtual ~ConcatenationKernel_fs_b_yx_fsv32() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const concatenation_params& params) const override;
    JitConstants GetJitConstants(const concatenation_params& params) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector
