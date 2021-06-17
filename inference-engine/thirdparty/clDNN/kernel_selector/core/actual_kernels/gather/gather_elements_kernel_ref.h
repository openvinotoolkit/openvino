/*
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
*/

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_elements_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gather_elements_params : public base_params {
    gather_elements_params() : base_params(KernelType::GATHER_ELEMENTS), indices_rank(0), batch_dims(0) {}

    uint8_t indices_rank;

    uint8_t batch_dims;
    uint8_t axis;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_elements_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gather_elements_optional_params : optional_params {
    gather_elements_optional_params() : optional_params(KernelType::GATHER_ELEMENTS) {}
};

class GatherElementsKernelRef : public KernelBaseOpenCL {
public:
    GatherElementsKernelRef() : KernelBaseOpenCL("gather_elements_ref") {}
    virtual ~GatherElementsKernelRef() {}
    virtual JitConstants GetJitConstants(const gather_elements_params& params) const;
    virtual CommonDispatchData SetDefault(const gather_elements_params& params, const optional_params&) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION,
                 FusedOpType::ELTWISE };
    }

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector
