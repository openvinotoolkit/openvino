/*
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
*/

#include "eltwise_kernel_ref.h"
#include "kernel_selector_utils.h" 

namespace kernel_selector {

    ParamsKey EltwiseKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputDataType(Datatype::INT8);
        k.EnableInputDataType(Datatype::INT32);
        k.EnableInputDataType(Datatype::INT64);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::INT8);
        k.EnableOutputDataType(Datatype::INT32);
        k.EnableOutputDataType(Datatype::INT64);
        k.EnableDifferentTypes();
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        k.EnableInt8Quantization();
        k.EnableOutputCalibration();
        return k;
    }

    bool EltwiseKernelRef::Validate(const Params& p, const optional_params& o) const
    {
        if (!EltwiseKernelBase::Validate(p, o))
        {
            return false;
        }

        const eltwise_params& params = static_cast<const eltwise_params&>(p);
        for (size_t i = 0; i < params.inputs.size(); i++)
        {
            if (params.inputs[i].GetLayout() == DataLayout::fs_bs_yx_bsv4_fsv32)
                return false;
        }
        if (params.output.GetLayout() == DataLayout::fs_bs_yx_bsv4_fsv32)
            return false;

        return true;
    }

    KernelsData EltwiseKernelRef::GetKernelsData(const Params& params, const optional_params& options) const
    {
        return GetCommonKernelsData(params, options);
    }
}