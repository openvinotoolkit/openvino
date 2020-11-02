// Copyright (c) 2020 Intel Corporation
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


#include "convolution_kernel_bfyx_iyxo.h"
#include <vector>
#include <utility>
#include <algorithm>

namespace kernel_selector {
// Sub-group size used by "convolution_kernel_bfyx_iyxo" kernel.
constexpr size_t sub_group_size = 16;

ParamsKey ConvolutionKernel_bfyx_iyxo::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableSubGroup();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_iyxo::SetDefault(const convolution_params& cp, int) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(cp);

    dispatchData.efficiency = FORCE_PRIORITY_9;

    dispatchData.gws[0] = CeilDiv(cp.output.X().v, sub_group_size) / 4;
    dispatchData.gws[1] = cp.output.Y().v;
    dispatchData.gws[2] = sub_group_size;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = sub_group_size;

    return dispatchData;
}

bool ConvolutionKernel_bfyx_iyxo::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o) || !CovolutionCheckInput(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);
    if (params.inputs[0].X().v % 64)
        return false;

    bool bFilterSize = (params.filterSize.x == 5 && params.filterSize.y == 5) ||
                       (params.filterSize.x == 3 && params.filterSize.y == 3 && (params.inputs[0].Feature().v % 4) == 0) ||
                       (params.filterSize.x == 1 && params.filterSize.y == 1);

    bool bStride = (params.stride.x == 1 && params.stride.y == 1);

    if (!bFilterSize || !bStride || (params.output.Feature().v % 4) != 0 || (params.output.Batch().v != 1)) {
        return false;
    }

    return true;
}

JitConstants ConvolutionKernel_bfyx_iyxo::GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", dispatchData.lws[2]));

    return jit;
}

KernelsData ConvolutionKernel_bfyx_iyxo::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

}  // namespace kernel_selector
