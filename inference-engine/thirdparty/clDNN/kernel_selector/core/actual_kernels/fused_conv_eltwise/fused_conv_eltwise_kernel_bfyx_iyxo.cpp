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


#include "fused_conv_eltwise_kernel_bfyx_iyxo.h"
#include <vector>
#include <utility>
#include <algorithm>

namespace kernel_selector {
constexpr size_t sub_group_size = 16;

fused_conv_eltwise_kernel_bfyx_iyxo::fused_conv_eltwise_kernel_bfyx_iyxo()
    : fused_conv_eltwise_kernel_base("fused_conv_eltwise_gpu_bfyx_iyxo") {
}

ParamsKey fused_conv_eltwise_kernel_bfyx_iyxo::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::image_2d_rgba);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableFusedConvEltwSplitSupport();
    k.EnableFusedConvEltwDilation();
    k.EnableFusedConvEltwTranspose();
    k.EnableFusedConvEltwiseRWOutOpt();
    k.EnableFusedConvEltwDepthToSpaceFusing();
    return k;
}

fused_conv_eltwise_kernel_base::DispatchData fused_conv_eltwise_kernel_bfyx_iyxo::SetDefault(
    const fused_conv_eltwise_params& cp,
    int) const {
    DispatchData dispatchData = fused_conv_eltwise_kernel_base::SetDefault(cp);

    dispatchData.efficiency = FORCE_PRIORITY_9;

    dispatchData.gws[0] = CeilDiv(cp.output.X().v, sub_group_size) / 4 / 2;
    dispatchData.gws[1] = cp.output.Y().v / 2;
    dispatchData.gws[2] = sub_group_size;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = sub_group_size;

    return dispatchData;
}

bool fused_conv_eltwise_kernel_bfyx_iyxo::Validate(const Params& p, const optional_params& o) const {
    if (!fused_conv_eltwise_kernel_base::Validate(p, o) || !FusedConvolutionEltwiseCheckInput(p, o)) {
        return false;
    }

    const auto& params = static_cast<const fused_conv_eltwise_params&>(p);
    if (params.inputs[0].X().v % 128 || params.inputs[0].Y().v % 2)
        return false;

    return true;
}

JitConstants fused_conv_eltwise_kernel_bfyx_iyxo::GetJitConstants(const fused_conv_eltwise_params& params,
                                                                  const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", dispatchData.lws[2]));
    return jit;
}

KernelsData fused_conv_eltwise_kernel_bfyx_iyxo::GetKernelsData(const Params& params,
                                                                        const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

}  // namespace kernel_selector
