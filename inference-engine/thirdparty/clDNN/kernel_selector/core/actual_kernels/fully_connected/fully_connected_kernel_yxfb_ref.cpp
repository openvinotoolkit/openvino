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

#include "fully_connected_kernel_yxfb_ref.h"

namespace kernel_selector 
{
    ParamsKey FullyConnected_yxfb_ref::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableAllInputLayout();
        k.EnableOutputLayout(DataLayout::fb);
        k.EnableBatching();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        return k;
    }

    KernelsData FullyConnected_yxfb_ref::GetKernelsData(const Params& params, const optional_params& options) const
    {
        KernelsData res = {};
        for (size_t i = 0; i < autoTuneOptions.size(); i++)
        {
            KernelsData kd = GetTunedKernelsDataByIndex(params, options, DataLayout::yxfb,
                { WeightsLayout::io, WeightsLayout::oi, WeightsLayout::oiyx, WeightsLayout::oyxi, WeightsLayout::iyxo, WeightsLayout::yxio }, DONT_USE_IF_HAVE_SOMETHING_ELSE, (int)i);
            if (!kd.empty())
            {
                res.emplace_back(kd[0]);
            }
        }
        return res;
    }
}