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
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableDepthwiseSeparableOpt();
    k.EnableGradient();
    return k;
}

CommonDispatchData DeconvolutionKernel_bfyx_opt::SetDefault(const deconvolution_params& params) const {
    DispatchData kd;

    kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;
    auto wg_size = 16;

    kd.gws0 = Align(params.output.X().v, wg_size * params.stride.x);
    kd.gws1 = params.output.Y().v;
    kd.gws2 = params.output.Batch().v * params.output.Feature().v;
    kd.lws0 = wg_size;
    kd.lws1 = 1;
    kd.lws2 = 1;
    kd.effiency = FORCE_PRIORITY_6;
    return kd;
}
}  // namespace kernel_selector