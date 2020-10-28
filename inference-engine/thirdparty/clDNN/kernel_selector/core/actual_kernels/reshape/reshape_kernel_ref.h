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

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reshape_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reshape_params : public base_params {
    reshape_params() : base_params(KernelType::RESHAPE) {}

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reshape_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reshape_optional_params : optional_params {
    reshape_optional_params() : optional_params(KernelType::RESHAPE) {}
};

class ReshapeKernelRef : public KernelBaseOpenCL {
public:
    ReshapeKernelRef() : KernelBaseOpenCL("reshape_ref") {}
    virtual ~ReshapeKernelRef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& op) const override;
};
}  // namespace kernel_selector
