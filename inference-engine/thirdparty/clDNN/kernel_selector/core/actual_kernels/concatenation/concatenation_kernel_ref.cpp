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

#include "concatenation_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector 
{

    ParamsKey ConcatenationKernelRef::GetSupportedKey() const
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
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        k.EnableConcatAxis(ConcatAxis::X);
        k.EnableConcatAxis(ConcatAxis::Y);
        k.EnableConcatAxis(ConcatAxis::FEATURE);
        k.EnableConcatAxis(ConcatAxis::BATCH);
        k.EnableConcatKernelPerInput();
        return k;
    }

    JitConstants ConcatenationKernelRef::GetJitConstants(const concatenation_params& params) const
    {
        auto cldnnJit = ConcatenationKernelBase::GetJitConstants(params);
        const concatenation_params& orgParams = static_cast<const concatenation_params&>(params);
        if (orgParams.inputs[0].Feature().v != 1)
        {
            cldnnJit.AddConstant(MakeJitConstant("CHECK_FEATURES", 1));
        }

        auto input_format = params.inputs[0].GetLayout();
        auto output_format = params.output.GetLayout();

        //default values when input_format = output_format
        std::vector<uint32_t> dim_index = { 0, 1, 2, 3 };

        //case for input == bfyx, output == yxfb and input == yxfb, output == bfyx
        if (input_format != output_format)
        {
            if (input_format == kernel_selector::Tensor::DataLayout::yxfb)
            {
                dim_index[0] = 2;
                dim_index[1] = 3;
                dim_index[2] = 1;
                dim_index[3] = 0;
            }
            else
            {
                dim_index[0] = 3;
                dim_index[1] = 2;
                dim_index[2] = 0;
                dim_index[3] = 1;
            }
        }

        cldnnJit.AddConstant(MakeJitConstant("INPUT_DIM_0", dim_index[0]));
        cldnnJit.AddConstant(MakeJitConstant("INPUT_DIM_1", dim_index[1]));
        cldnnJit.AddConstant(MakeJitConstant("INPUT_DIM_2", dim_index[2]));
        cldnnJit.AddConstant(MakeJitConstant("INPUT_DIM_3", dim_index[3]));

        return cldnnJit;
    }

    KernelsData ConcatenationKernelRef::GetKernelsData(const Params& params, const optional_params& optParams) const
    {
        KernelsData kd = GetCommonKernelsData(params, optParams);

        if (!kd.empty())
        {
            for (int i = 0; i < (int)kd[0].kernels.size(); i++)
            {
                auto& kernel = kd[0].kernels[i];

                // to avoid cases when we execute with local work sizes 1x1x1
                if (kernel.workGroups.local[0] == 1 &&
                    kernel.workGroups.global[1] != 1)
                {
                    kernel.workGroups.global[1] = Align(kernel.workGroups.global[1], 32);
                    kernel.workGroups.local[1] = 32;
                }
            }
        }

        return kd;
     }
}