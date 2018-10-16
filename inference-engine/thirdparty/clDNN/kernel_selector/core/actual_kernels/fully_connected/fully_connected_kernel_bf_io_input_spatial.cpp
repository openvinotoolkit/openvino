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

#include "fully_connected_kernel_bf_io_input_spatial.h"
#include "kernel_selector_utils.h"

namespace kernel_selector 
{
    ParamsKey FullyConnected_bf_io_input_spatial::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableAllInputLayout();
        k.EnableOutputLayout(DataLayout::bf);
        k.EnableBatching();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        return k;
    }

    std::unique_ptr<FullyConnected_bf_io_input_spatial::DispatchData> FullyConnected_bf_io_input_spatial::SetDefault(const fully_connected_params& arg) const
    {
        auto kd = FullyConnectedKernelBase::SetDefault(arg);

        kd->gws0 = Align(arg.output.LogicalSize() / arg.inputs[0].Batch().v, 16);
        kd->gws1 = arg.inputs[0].Batch().v;
        kd->gws2 = 1;
        kd->lws0 = 16;
        kd->lws1 = 1;
        kd->lws2 = 1;

        kd->effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        const auto &input = arg.inputs[0];
        const auto &output = arg.output;

        if (input.Batch().v == 1 && output.Batch().v == 1)
        {
            if ((input.LogicalSize() / output.Batch().v >= 9216) && (output.Feature().v >= 4096))
            {
                kd->effiency = FORCE_PRIORITY_1;
            }
        }

        return std::move(kd);
    }

    bool FullyConnected_bf_io_input_spatial::Validate(const Params& p, const optional_params& o) const
    {
        if (!FullyConnectedKernelBase::Validate(p, o))
        {
            return false;
        }

        const auto& params = static_cast<const fully_connected_params&>(p);

        const auto &input = params.inputs[0];
        const auto &output = params.output;
        if ( (input.GetLayout() != DataLayout::bfyx && input.GetLayout() != DataLayout::bf) ||
             (output.GetLayout() != DataLayout::bfyx && output.GetLayout() != DataLayout::bf))
        {
            return false;
        }

        return true;
    }

    KernelsData FullyConnected_bf_io_input_spatial::GetKernelsData(const Params& params, const optional_params& optParams) const
    {
        const auto& orgParams = static_cast<const fully_connected_params&>(params);

        const auto& input = orgParams.inputs[0];
        const auto& output = orgParams.output;

        if (input.GetLayout() == DataLayout::bfyx)
        {
            if (input.Batch().v == 1 && output.Batch().v == 1)
            {
                if ((input.LogicalSize() / output.Batch().v >= 9216) && (output.Feature().v >= 4096))
                {
                    return GetCommonKernelsData(params, optParams, DataLayout::bf, { WeightsLayout::io }, FORCE_PRIORITY_1);
                }
            }
        }
        return GetCommonKernelsData(params, optParams, DataLayout::bf, { WeightsLayout::io });
    }
}