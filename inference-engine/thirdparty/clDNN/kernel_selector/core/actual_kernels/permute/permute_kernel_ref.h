// Copyright (c) 2016-2021 Intel Corporation
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

#include "permute_kernel_base.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PermuteKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class PermuteKernelRef : public PermuteKernelBase {
public:
    using Parent = PermuteKernelBase;
    using Parent::Parent;
    PermuteKernelRef() : PermuteKernelBase("permute_ref") {}
    virtual ~PermuteKernelRef() {}

    bool Validate(const Params& p, const optional_params& o) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const override;
    CommonDispatchData SetDefault(const permute_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE,
            FusedOpType::SCALE
        };
    }
};
}  // namespace kernel_selector
