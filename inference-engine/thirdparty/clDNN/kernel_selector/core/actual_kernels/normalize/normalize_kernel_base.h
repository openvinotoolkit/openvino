// Copyright (c) 2016-2020 Intel Corporation
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

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// normalize_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct normalize_params : public base_params {
    normalize_params() : base_params(KernelType::NORMALIZE) {}

    NormalizeMode normMode = NormalizeMode::ACROSS_SPATIAL;
    float epsilon = 1e-10f;
    DataTensor scaleTable;

    virtual ParamsKey GetParamsKey() const {
        ParamsKey k = base_params::GetParamsKey();

        k.EnableNormalizeMode(normMode);

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// normalize_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct normalize_optional_params : optional_params {
    normalize_optional_params() : optional_params(KernelType::NORMALIZE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NormalizeKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class NormalizeKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~NormalizeKernelBase() {}

    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const normalize_params& params) const;
    DispatchData SetDefault(const normalize_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimated_time) const;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION,
                 FusedOpType::SCALE };
    }
    bool Validate(const Params& params, const optional_params&) const override;
    Datatype GetActivationType(const normalize_params& params) const;
};
}  // namespace kernel_selector
