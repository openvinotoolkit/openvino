// Copyright (C) 2018-2025 Intel Corporation
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
class EltwiseKernel_blocked_opt : public EltwiseKernelBase {
public:
    EltwiseKernel_blocked_opt() : EltwiseKernelBase("eltwise_blocked_opt") {}
    virtual ~EltwiseKernel_blocked_opt() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::QUANTIZE,
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE
        };
    }

protected:
    bool Validate(const Params& p) const override;
    JitConstants MakeLoadJitConstants(const eltwise_params& params, bool useVload8) const override;
    JitConstants GetJitConstants(const eltwise_params& params) const override;
    DispatchData SetDefault(const eltwise_params& params) const override;
    void PrintWorkSize(const DispatchData& dis);
};
}  // namespace kernel_selector
