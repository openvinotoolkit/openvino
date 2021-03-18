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


#pragma once

#include "mvn_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {
class MVNKernel_b_fs_yx_fsv16_imad : public MVNKernelBase {
public:
    using Parent = MVNKernelBase;
    MVNKernel_b_fs_yx_fsv16_imad() : MVNKernelBase("mvn_gpu_b_fs_yx_fsv16_imad") {}
    virtual ~MVNKernel_b_fs_yx_fsv16_imad() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    struct MultiDispatchData {
        DispatchData stage_1;
        DispatchData stage_2;
        DispatchData stage_final;

        size_t item_groups;
    };

    bool Validate(const Params&, const optional_params&) const override;
    DispatchData SetDefault(const mvn_params& params) const override;
    JitConstants GetJitConstants(const mvn_params& params, DispatchData dispatchData) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE,
            FusedOpType::SCALE
        };
    }

    KernelsData GetMultiStageKernelsData(const mvn_params& params, const optional_params&) const;
    MultiDispatchData SetDefaultForMulti(const mvn_params& params) const;
};
}  // namespace kernel_selector
