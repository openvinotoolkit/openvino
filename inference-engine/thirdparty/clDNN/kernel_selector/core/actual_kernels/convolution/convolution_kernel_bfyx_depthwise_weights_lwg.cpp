// Copyright (c) 2018-2020 Intel Corporation
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


#include "convolution_kernel_bfyx_depthwise_weights_lwg.h"
#include <vector>

namespace kernel_selector {
ParamsKey ConvolutionKernel_bfyx_depthwise_weights_lwg::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    k.EnableDepthwiseSeparableOpt();
    k.EnableDilation();
    k.EnableGroupedConvolution();
    return k;
}

bool ConvolutionKernel_bfyx_depthwise_weights_lwg::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o) || !CovolutionCheckInput(p, o)) {
        return false;
    }

    const convolution_params& cp = static_cast<const convolution_params&>(p);

    if ((cp.filterSize.x > 5) || (cp.filterSize.y > 5) || (cp.groups == 1) ||
        (cp.weights.IFM().v != 1) || (cp.weights.OFM().v != 1)) {
        return false;
    }

    return true;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_depthwise_weights_lwg::SetDefault(const convolution_params& params,
                                                                                             int) const {
    DispatchData dispatchData = Parent::SetDefault(params);
    const auto& out = params.output;

    dispatchData.gws = { Align(out.X().v * out.Y().v, 16), out.Feature().v, out.Batch().v };
    dispatchData.lws = { 16, 1, 1 };

    dispatchData.efficiency = FORCE_PRIORITY_2;

    return dispatchData;
}

JitConstants ConvolutionKernel_bfyx_depthwise_weights_lwg::GetJitConstants(const convolution_params& params,
                                                                           const DispatchData& dispatchData) const {
    auto mem_consts = ConvolutionKernelBase::GetJitConstants(params, dispatchData);

    if (params.padding.x != 0 || params.padding.y != 0)
        mem_consts.AddConstant(MakeJitConstant("BOUNDARY_CHECK", 1));

    return mem_consts;
}

KernelsData ConvolutionKernel_bfyx_depthwise_weights_lwg::GetKernelsData(const Params& params,
                                                                         const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector
