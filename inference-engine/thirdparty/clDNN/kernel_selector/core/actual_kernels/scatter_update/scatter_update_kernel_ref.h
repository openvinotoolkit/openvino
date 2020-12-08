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

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// scatter_update_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct scatter_update_params : public base_params {
    scatter_update_params() : base_params(KernelType::SCATTER_UPDATE), axis(ScatterUpdateAxis::BATCH) {}

    ScatterUpdateAxis axis;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// scatter_update_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct scatter_update_optional_params : optional_params {
    scatter_update_optional_params() : optional_params(KernelType::SCATTER_UPDATE) {}
};

class ScatterUpdateKernelRef : public KernelBaseOpenCL {
public:
    ScatterUpdateKernelRef() : KernelBaseOpenCL("scatter_update_ref") {}
    virtual ~ScatterUpdateKernelRef() {}
    virtual JitConstants GetJitConstants(const scatter_update_params& params) const;
    virtual CommonDispatchData SetDefault(const scatter_update_params& params, const optional_params&, bool is_second) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector
