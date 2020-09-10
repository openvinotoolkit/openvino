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

#include "fully_connected_kernel_base.h"
#include <vector>

namespace kernel_selector {

class FullyConnectedKernelMMAD : public FullyConnectedKernelBase {
public:
    using Parent = FullyConnectedKernelBase;

    FullyConnectedKernelMMAD() : Parent("fully_connected_gpu_MMAD") {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

    struct FullyConnectedTuningData {
        const size_t sub_group_size = 8;
        size_t slm_div_factor = 1;
        size_t work_group_size = 1;
    };

protected:
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& kd) const override;
    DispatchData SetDefault(const fully_connected_params& params, int autoTuneIndex = -1) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }
    bool Validate(const Params& params, const optional_params& options) const override;
    FullyConnectedTuningData SetTuningParams(const fully_connected_params& params) const;
};
}  // namespace kernel_selector
