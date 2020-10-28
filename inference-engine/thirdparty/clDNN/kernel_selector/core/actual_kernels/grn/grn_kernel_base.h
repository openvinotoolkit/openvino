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

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <string>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// grn_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct grn_params : public base_params {
    grn_params() : base_params(KernelType::GRN) {}

    float bias = 1.0f;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// grn_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct grn_optional_params : optional_params {
    grn_optional_params() : optional_params(KernelType::GRN) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GRNKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class GRNKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~GRNKernelBase() {}
    using DispatchData = CommonDispatchData;

protected:
    virtual JitConstants GetJitConstants(const grn_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const grn_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimated_time) const;
};
}  // namespace kernel_selector
