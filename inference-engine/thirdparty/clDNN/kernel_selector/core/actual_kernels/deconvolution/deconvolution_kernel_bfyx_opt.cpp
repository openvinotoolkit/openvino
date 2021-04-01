// Copyright (c) 2016-2020 Intel Corporation
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


#include "deconvolution_kernel_bfyx_opt.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey DeconvolutionKernel_bfyx_opt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableDepthwiseSeparableOpt();
    k.EnableGroupedConvolution();
    k.EnableDifferentTypes();
    return k;
}

CommonDispatchData DeconvolutionKernel_bfyx_opt::SetDefault(const deconvolution_params& params) const {
    DispatchData dispatchData;

    auto wg_size = 16;

    dispatchData.gws[0] = Align(params.output.X().v, wg_size * params.stride.x);
    dispatchData.gws[1] = params.output.Y().v;
    dispatchData.gws[2] = params.output.Batch().v * params.output.Feature().v;

    dispatchData.lws[0] = wg_size;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority DeconvolutionKernel_bfyx_opt::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_6;
}

JitConstants DeconvolutionKernel_bfyx_opt::GetJitConstants(const deconvolution_params& params) const {
    auto jit = Parent::GetJitConstants(params);

    if (!params.fused_ops.empty()) {
        auto fused_dt = GetActivationType(params);
        FusedOpsConfiguration conf = {
            "",
            {"batch_offset", "ofm_offset", "id_y", "id_x"},
            "result",
            fused_dt,
            1,
            LoadType::LT_UNALIGNED,
            BoundaryCheck::DISABLED };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }
    return jit;
}

}  // namespace kernel_selector
