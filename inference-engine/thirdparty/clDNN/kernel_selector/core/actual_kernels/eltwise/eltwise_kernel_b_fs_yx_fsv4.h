// Copyright (c) 2021 Intel Corporation
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

#include "eltwise_kernel_base.h"

namespace kernel_selector {
class EltwiseKernel_b_fs_yx_fsv4 : public EltwiseKernelBase {
public:
    EltwiseKernel_b_fs_yx_fsv4() : EltwiseKernelBase("eltwise_b_fs_yx_fsv4") {}
    virtual ~EltwiseKernel_b_fs_yx_fsv4() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::QUANTIZE,
            FusedOpType::ACTIVATION,
            FusedOpType::SCALE,
            FusedOpType::ELTWISE
        };
    }

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants MakeLoadJitConstants(const eltwise_params& params, bool useVload8) const override;
    JitConstants GetJitConstants(const eltwise_params& params) const override;
    DispatchData SetDefault(const eltwise_params& params) const override;
    void PrintWorkSize(const DispatchData& dis);

    const int vec_size = 4;
};
}  // namespace kernel_selector