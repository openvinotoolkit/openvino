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

#include "upsampling_kernel_ref.h"

namespace kernel_selector {

    ParamsKey UpSamplingKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableDifferentTypes();
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        k.EnableUpSamplingSampleType(SampleType::NEAREST);
        return k;
    }

    KernelsData UpSamplingKernelRef::GetKernelsData(const Params& params, const optional_params& options) const
    {
        return GetCommonKernelsData(params, options);
    }
}