/*
// Copyright (c) 2020 Intel Corporation
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

#include "space_to_batch_kernel_base.h"

namespace kernel_selector {
class SpaceToBatchKernelRef : public SpaceToBatchKernelBase {
public:
    using Parent = SpaceToBatchKernelBase;
    SpaceToBatchKernelRef() : SpaceToBatchKernelBase("space_to_batch_ref") {}
    virtual ~SpaceToBatchKernelRef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const space_to_batch_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::ELTWISE,
            FusedOpType::SCALE,
            FusedOpType::QUANTIZE
        };
    }
};
}  // namespace kernel_selector
