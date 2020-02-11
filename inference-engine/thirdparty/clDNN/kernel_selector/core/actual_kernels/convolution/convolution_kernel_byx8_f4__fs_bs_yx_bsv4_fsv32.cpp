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

#include "convolution_kernel_byx8_f4__fs_bs_yx_bsv4_fsv32.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey ConvolutionKernel_byx8_f4__fs_bs_yx_bsv4_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::byx8_f4);
    k.EnableOutputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.DisableTuning();
    return k;
}

bool ConvolutionKernel_byx8_f4__fs_bs_yx_bsv4_fsv32::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o)) {
        return false;
    }

    return true;
}

size_t static get_wg_batch_size(const convolution_params& params) {
    if (params.inputs[0].Batch().v % 64 == 0)
        return 32;
    return 1;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_byx8_f4__fs_bs_yx_bsv4_fsv32::SetDefault(
    const convolution_params& arg,
    int) const {
    DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

    runInfo.effiency = FORCE_PRIORITY_1;

    runInfo.gws0 = (arg.output.Batch().v * arg.output.Feature().v) / (4 * 2);
    runInfo.gws1 = arg.output.X().v / 8;
    runInfo.gws2 = arg.output.Y().v / 2;

    runInfo.lws0 = 8 * get_wg_batch_size(arg);
    runInfo.lws1 = 1;
    runInfo.lws2 = 1;

    return runInfo;
}

JitConstants ConvolutionKernel_byx8_f4__fs_bs_yx_bsv4_fsv32::GetJitConstants(const convolution_params& params,
                                                                             const DispatchData& kd) const {
    auto jits = ConvolutionKernelBase::GetJitConstants(params, kd);

    jits.AddConstant(MakeJitConstant("WG_BATCH_SIZE", get_wg_batch_size(params)));

    return jits;
}

KernelsData ConvolutionKernel_byx8_f4__fs_bs_yx_bsv4_fsv32::GetKernelsData(const Params& params,
                                                                           const optional_params& options) const {
    KernelsData kd = GetCommonKernelsData(params, options, " -Dcl_intel_subgroups_char");
    if (!kd.empty())
        kd[0].estimatedTime = FORCE_PRIORITY_3;
    return kd;
}
}  // namespace kernel_selector
