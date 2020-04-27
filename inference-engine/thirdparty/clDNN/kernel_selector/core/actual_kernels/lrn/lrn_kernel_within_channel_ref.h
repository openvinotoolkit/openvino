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

#include "lrn_kernel_base.h"
#include "vector"

namespace kernel_selector {
class LRNKernelWithinChannel : public LRNKernelBase {
public:
    using Parent = LRNKernelBase;
    LRNKernelWithinChannel() : LRNKernelBase("lrn_gpu_within_channel") {}
    virtual ~LRNKernelWithinChannel() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

private:
    DispatchData SetDefault(const lrn_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }
    bool Validate(const Params& params, const optional_params& options) const override;
    JitConstants GetJitConstants(const lrn_params& params, const DispatchData& kd) const override;
};
}  // namespace kernel_selector
