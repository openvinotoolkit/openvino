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

#include "fully_connected_kernel_fb_io_b8_f8.h"

namespace kernel_selector 
{
    ParamsKey FullyConnected_fb_io_b8_f8::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableAllInputLayout();
        k.EnableOutputLayout(DataLayout::fb);
        k.EnableBatching();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableSubGroup();
        return k;
    }

    FullyConnected_fb_io_b8_f8::DispatchData FullyConnected_fb_io_b8_f8::SetDefault(const fully_connected_params& arg, int ) const
    {
        auto kd = FullyConnectedBlockKernelBase::SetDefault(arg);

        const auto& output = arg.output;
        
        size_t groups_per_batches = GetLocalGroupsSize(arg);
        kd.gws0 = Align(output.LogicalSize() / (GetNeuronsPerWorkItem(arg) * GetBatchesPerWorkItem(arg) * groups_per_batches), 8);
        kd.gws1 = groups_per_batches;
        kd.lws0 = 8;
        kd.lws1 = 1;

        return kd;
    }

    bool FullyConnected_fb_io_b8_f8::Validate(const Params& p, const optional_params& o) const
    {
        if (!FullyConnectedBlockKernelBase::Validate(p, o))
        {
            return false;
        }

        const auto& params = static_cast<const fully_connected_params&>(p);

        const auto& output = params.output;
        const auto batches = output.Batch().v;
        const auto x_size = output.LogicalSize() / batches;

        const auto& input = params.inputs[0];
        const auto input_x_size = input.LogicalSize() / input.Batch().v;
        const bool proper_input_aligment = (input_x_size % 8) == 0;
        const bool proper_output_aligment = (output.LogicalSize() / (GetNeuronsPerWorkItem(params) * GetBatchesPerWorkItem(params) * GetLocalGroupsSize(params)) % 8) == 0;
        const bool bSupportedBatch = (batches % 8) == 0;
        const bool bSupportedFeature = (x_size % 8) == 0;

        if (!bSupportedBatch ||
            !bSupportedFeature ||
            !proper_input_aligment ||
            !proper_output_aligment)
        {
            return false;
        }

        return true;
    }

    KernelsData FullyConnected_fb_io_b8_f8::GetKernelsData(const Params& params, const optional_params& optParams) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);
        KernelsData res = {};
        const auto& orgParams = static_cast<const fully_connected_params&>(params);

        float estimated_time =
            orgParams.inputs[0].GetDType() == Datatype::F16 && orgParams.output.Batch().v >= 16 ?
            FORCE_PRIORITY_3 : FORCE_PRIORITY_5;
        
        for (size_t i = 0; i < autoTuneOptions.size(); i++)
        {
            KernelsData kd = GetTunedKernelsDataByIndex(params, optParams, DataLayout::fb, { WeightsLayout::io }, estimated_time, (int)i);
            if (!kd.empty())
            {
                res.emplace_back(kd[0]);
            }
        }

        return res;
    }
}
