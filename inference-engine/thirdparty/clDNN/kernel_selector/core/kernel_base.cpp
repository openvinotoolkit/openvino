// Copyright (c) 2016-2019 Intel Corporation
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


#include "kernel_base.h"

namespace kernel_selector {
const primitive_db KernelBase::db;
size_t KernelBase::counter = 0;

static bool IsTypeUsedIn(Datatype type, const base_params& params) {
    return params.output.GetDType() == type ||
           std::any_of(params.inputs.begin(), params.inputs.end(), [=](const DataTensor& input) -> bool {
               return input.GetDType() == type;
           });
}

Datatype KernelBase::GetUnitType(const base_params& params) const {
    Datatype types_prioritized[] =
        {Datatype::INT8, Datatype::F16, Datatype::INT32, Datatype::INT64, Datatype::UINT8, Datatype::UINT32};

    for (Datatype type : types_prioritized)
        if (IsTypeUsedIn(type, params))
            return type;

    return Datatype::F32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeBaseParamsJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
JitConstants KernelBase::MakeBaseParamsJitConstants(const base_params& params) const {
    auto unitType = GetUnitType(params);

    JitConstants jit{
        MakeJitConstant("OUTPUT", params.output),
        MakeJitConstant("FP64_SUPPORTED", params.engineInfo.bFP64Support),
        MakeJitConstant("FP16_SUPPORTED", params.engineInfo.bFP16Support),
        MakeJitConstant("FP16_UNIT_USED", IsTypeUsedIn(Datatype::F16, params)),
        MakeJitConstant("INT8_UNIT_USED", IsTypeUsedIn(Datatype::INT8, params)),
        MakeJitConstant("INT32_UNIT_USED", IsTypeUsedIn(Datatype::INT32, params)),
        MakeJitConstant("INT64_UNIT_USED", IsTypeUsedIn(Datatype::INT64, params)),
        MakeJitConstant("UINT8_UNIT_USED", IsTypeUsedIn(Datatype::UINT8, params)),
        MakeJitConstant("UINT32_UNIT_USED", IsTypeUsedIn(Datatype::UINT32, params)),
        MakeJitConstant("GRADIENT", params.gradient),
    };

    // for activation function
    jit.Merge(MakeUnitTypeJitConstants(unitType));
    jit.Merge(MakeActivationJitConstants(params.activation));

    for (size_t i = 0; i < params.inputs.size(); i++) {
        jit.AddConstant(MakeJitConstant("INPUT" + toCodeString(i), params.inputs[i]));
    }

    jit.AddConstant(MakeJitConstant("LayerID", params.layerID));

    return jit;
}

}  // namespace kernel_selector
