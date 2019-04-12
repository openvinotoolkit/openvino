/*
// Copyright (c) 2018 Intel Corporation
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

#include "select_kernel_base.h"
#include "kernel_selector_utils.h" 

namespace kernel_selector
{
    
    bool SelectKernelBase::Validate(const Params& p, const optional_params& o) const
    {
        if (p.GetType() != KernelType::SELECT ||
            o.GetType() != KernelType::SELECT)
        {
            return false;
        }

        const select_params& params = static_cast<const select_params&>(p);

		if (params.inputs[0].GetDType() != params.inputs[1].GetDType()) 
		{
			return false;
		}

        if (params.inputs.size() != 3)
        {
            return false;
        }

        return true;
    }

    JitConstants SelectKernelBase::GetJitConstantsCommon(const select_params& params) const
    {
        JitConstants jit = MakeBaseParamsJitConstants(params);

        std::string inputs_decls;

        for (size_t i = 0; i < params.inputs.size(); i++)
        {
            std::string const_str = "const";

            inputs_decls += const_str + " __global " + toCLType(params.inputs[i].GetDType()) + "* input" + std::to_string(i) + ", ";
        }

        jit.AddConstant(MakeJitConstant("INPUTS_DECLS", inputs_decls));

		std::string destType, absType;

		// i8, i8, i8
		// i8, i8, u8
		// u8, u8, i8
		// u8, u8, u8
		if ((params.inputs[2].GetDType() == Datatype::INT8
			|| params.inputs[2].GetDType() == Datatype::UINT8)
			&& (params.inputs[0].GetDType() == Datatype::INT8
				|| params.inputs[0].GetDType() == Datatype::UINT8))
		{
			jit.AddConstant(MakeJitConstant("MASK", "INPUT_2"));
		}
		else
		{
			// x, x, f32
			// x, x, f16
			if (params.inputs[2].GetDType() == Datatype::F32
				|| params.inputs[2].GetDType() == Datatype::F16)
			{
				absType = "fabs";
			}
			// f32, f32, i8
			// f32, f32, u8
			// f16, f16, i8
			// f16, f16, u8
			else
			{
				absType = "abs";
			}

			// f32, f32, x
			if (params.inputs[0].GetDType() == Datatype::F32) {
				destType = "int";
			}
			// f16, f16, x
			else if (params.inputs[0].GetDType() == Datatype::F16) {
				destType = "short";
			}
			// i8, i8, f32
			// i8, i8, f16
			// u8, u8, f32
			// u8, u8, f16
			else
			{
				destType = "char";
			}

			jit.AddConstant(MakeJitConstant("MASK", "convert_" + destType + "_rtp(" + absType + "(INPUT_2))"));
		}

        return jit;
    }

    JitConstants SelectKernelBase::GetJitConstants(const select_params& params) const
    {
        return GetJitConstantsCommon(params);
    }

    SelectKernelBase::DispatchData SelectKernelBase::SetDefault(const select_params& params) const
    {
        DispatchData kd;

        const auto& out = params.output;

        std::vector<size_t> gws;
        for (const auto& o : out.GetDims())
        {
            gws.push_back(o.v);
        }

        for (size_t i = gws.size(); i < 4; i++)
        {
            gws.push_back(1U);
        }

        kd.gws0 = gws[0];
        kd.gws1 = gws[1];
        kd.gws2 = gws[2] * gws[3];

        auto local = GetOptimalLocalWorkGroupSizes( { kd.gws0, kd.gws1, kd.gws2 } );
        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];

        return kd;
    }

    KernelsData SelectKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        KernelData kd = KernelData::Default<select_params>(params);
        select_params& newParams = *static_cast<select_params*>(kd.params.get());

        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto cldnn_jit = GetJitConstants(newParams);
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        DispatchData runInfo = SetDefault(newParams);

        auto& kernel = kd.kernels[0];

        kernel.workGroups.global = { runInfo.gws0, runInfo.gws1, runInfo.gws2 };
        kernel.workGroups.local = { runInfo.lws0, runInfo.lws1, runInfo.lws2 };

        kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
        kernel.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(), false, false);

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}
