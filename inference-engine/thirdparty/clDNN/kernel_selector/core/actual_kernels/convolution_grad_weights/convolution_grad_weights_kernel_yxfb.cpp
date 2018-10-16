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

#include "convolution_grad_weights_kernel_yxfb.h"

namespace kernel_selector
{

	ParamsKey ConvolutionGradWeightsKernel_yxfb::GetSupportedKey() const
	{
		ParamsKey k;
		k.EnableInputDataType(Datatype::F32);
		k.EnableInputWeightsType(WeightsType::F32);
		k.EnableOutputDataType(Datatype::F16);
		k.EnableOutputDataType(Datatype::F32);
		k.EnableInputLayout(DataLayout::yxfb);
		k.EnableOutputLayout(DataLayout::yxfb);
		k.EnableOutputLayout(DataLayout::bfyx);
		k.EnableOutputLayout(DataLayout::byxf);
		k.EnableSubGroup();
		k.EnableTensorOffset();
		k.EnableTensorPitches();
		k.EnableBiasPerFeature();
		k.EnableNonBiasTerm();
		k.EnableMomentum();
		k.EnableBatching();
		k.EnableSplitSupport();
		k.EnableGradient();
		k.DisableTuning();
		return k;
	}

	bool ConvolutionGradWeightsKernel_yxfb::Validate(const Params& p, const optional_params&) const
	{
		const convolution_grad_weights_params& params = static_cast<const convolution_grad_weights_params&>(p);
		auto batch = params.inputs[0].Batch().v;

		if (batch % 16 != 0)
			return false;
		if (params.stride.x != 1 || params.stride.y != 1)
			return false;
		return true;
	}

	ConvolutionGradWeightsKernelBase::DispatchData ConvolutionGradWeightsKernel_yxfb::SetDefault(const convolution_grad_weights_params& params) const
	{
		auto input_features = params.weights.IFM().v;
		auto output_features = params.weights.OFM().v;
		auto x = params.weights.X().v;
		auto y = params.weights.Y().v;

		DispatchData kd;

		kd.gws0 = 32;
		kd.gws1 = input_features * output_features;
		kd.gws2 = x * y;

		kd.lws0 = 32;
		kd.lws1 = 1;
		kd.lws2 = 1;
		kd.effiency = FORCE_PRIORITY_7;

		return kd;
	}
}