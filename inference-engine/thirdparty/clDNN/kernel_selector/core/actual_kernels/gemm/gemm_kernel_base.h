// Copyright (c) 2018-2020 Intel Corporation
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

#include "common_kernel_base.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gemm_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gemm_params : public base_params {
    gemm_params()
        : base_params(KernelType::GEMM), alpha(1.0f), beta(0.0f), transpose_input0(false), transpose_input1(false) {}

    float alpha;
    float beta;
    bool transpose_input0;
    bool transpose_input1;
    QuantizationType quantization = QuantizationType::NONE;

    virtual ParamsKey GetParamsKey() const {
        ParamsKey k = base_params::GetParamsKey();
        k.EnableQuantization(quantization);
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gemm_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gemm_optional_params : optional_params {
    gemm_optional_params() : optional_params(KernelType::GEMM) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BorderKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class GemmKernelBase : public common_kernel_base {
public:
    using common_kernel_base::common_kernel_base;
    using FusedOpDesc = fused_operation_desc;
    using DispatchData = CommonDispatchData;
    virtual ~GemmKernelBase() {}

protected:
    virtual JitConstants GetJitConstants(const gemm_params& params) const;
    virtual DispatchData SetDefault(const gemm_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimated_time) const;
    // Fused ops
    virtual JitConstants GetFusedPrimitivesJitConstants(const gemm_params& params, const DispatchData& dispatchData) const;
    Datatype GetActivationType(const gemm_params& params) const;
    // --Fused ops

    bool Validate(const Params& p, const optional_params&) const override;
};
}  // namespace kernel_selector
