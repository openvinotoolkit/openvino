/*
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
*/

#include "convolution_kernel_byxf_fs_bs_yx_bsv4_fsv32.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey ConvolutionKernel_byxf_fs_bs_yx_bsv4_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.DisableTuning();
    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_byxf_fs_bs_yx_bsv4_fsv32::SetDefault(
    const convolution_params& arg,
    int) const {
    DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

    runInfo.efficiency = FORCE_PRIORITY_1;

    runInfo.gws0 = (arg.output.Batch().v * arg.output.Feature().v) / 4;
    runInfo.gws1 = arg.output.X().v / 8;
    runInfo.gws2 = arg.output.Y().v;

    runInfo.lws0 = 8;
    runInfo.lws1 = 1;
    runInfo.lws2 = 1;

    return runInfo;
}

KernelsData ConvolutionKernel_byxf_fs_bs_yx_bsv4_fsv32::GetKernelsData(const Params& params,
                                                                       const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
}  // namespace kernel_selector
