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

#include "embed_kernel_ref.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace kernel_selector
{

	ParamsKey EmbedKernelRef::GetSupportedKey() const
	{
		ParamsKey k;
		k.EnableInputDataType(Datatype::F16);
		k.EnableInputDataType(Datatype::F32);
		k.EnableInputDataType(Datatype::INT8);
		k.EnableOutputDataType(Datatype::F16);
		k.EnableOutputDataType(Datatype::F32);
		k.EnableOutputDataType(Datatype::INT8);
		k.EnableInputWeightsType(WeightsType::F16);
		k.EnableInputWeightsType(WeightsType::F32);
		k.EnableInputWeightsType(WeightsType::INT8);
        k.EnableAllInputLayout();
		k.EnableOutputLayout(DataLayout::bf);
		k.EnableBiasPerOutput();
		k.EnableBiasPerFeature();
		k.EnableTensorOffset();
		k.EnableTensorPitches();
		k.EnableBatching();
        k.EnableNonBiasTerm();
		return k;
	}

	JitConstants EmbedKernelRef::GetJitConstants(const embed_params& params) const
	{
        JitConstants jit = WeightBiasKernelBase::GetJitConstants(params);
        const auto& input = params.inputs[0];
        const auto x_size = input.LogicalSize() / input.Batch().v;
        const auto w_size = params.weights.OFM().v;
        jit.AddConstant(MakeJitConstant("INPUT0_ELEMENTS_COUNT", x_size));
        jit.AddConstant(MakeJitConstant("NUM_OUTPUT_SIZE", w_size));

        return jit;
	}

	EmbedKernelRef::DispatchData EmbedKernelRef::SetDefault(const embed_params& params) const
	{
		DispatchData kd;
		std::vector<size_t> global = { params.inputs[0].X().v , params.weights.OFM().v, params.inputs[0].Batch().v };
		std::vector<size_t> local = GetOptimalLocalWorkGroupSizes(global);

		kd.gws0 = global[0];
		kd.gws1 = global[1];
		kd.gws2 = global[2];

		kd.lws0 = local[0];
		kd.lws1 = local[1];
		kd.lws2 = 1;
		return kd;

	}

	KernelsData EmbedKernelRef::GetKernelsData(const Params& params, const optional_params& options) const
	{
		assert(params.GetType() == KernelType::EMBED);

		const embed_params& orgParams = static_cast<const embed_params&>(params);

		const std::vector<WeightsLayout> weightsLayouts = {
			WeightsLayout::oiyx,
		};

		DispatchData runInfo = SetDefault(orgParams);
		KernelData kd = KernelData::Default<embed_params>(params);
		embed_params& newParams = *static_cast<embed_params*>(kd.params.get());

		bool succeed = UpdateWeightsParams(
			newParams,
			options,
			weightsLayouts,
			kd.weightsReorderParams);

		if (!succeed)
		{
			return{};
		}

		auto cldnn_jit = GetJitConstants(newParams);
		auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
		auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

		auto& kernel = kd.kernels[0];

		FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point, DEFAULT, true, !newParams.bias.empty());

		kd.estimatedTime = runInfo.effiency;

		return{ kd };
	}

	
}