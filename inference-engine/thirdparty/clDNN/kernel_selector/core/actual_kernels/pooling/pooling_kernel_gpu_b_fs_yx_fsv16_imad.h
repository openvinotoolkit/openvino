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

#include "pooling_kernel_base.h"
#include <vector>

namespace kernel_selector {
class PoolingKernelGPU_b_fs_yx_fsv16_imad: public PoolingKernelBase{
public:
    PoolingKernelGPU_b_fs_yx_fsv16_imad() : PoolingKernelBase("pooling_gpu_b_fs_yx_fsv16_imad") {}
    virtual ~PoolingKernelGPU_b_fs_yx_fsv16_imad() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const pooling_params& params) const override;
    bool Validate(const Params&, const optional_params&) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
            FusedOpType::QUANTIZE,
            FusedOpType::SCALE,
            FusedOpType::ACTIVATION };
    }

protected:
    JitConstants GetJitConstants(const pooling_params& params, DispatchData kd) const override;
};
}  // namespace kernel_selector