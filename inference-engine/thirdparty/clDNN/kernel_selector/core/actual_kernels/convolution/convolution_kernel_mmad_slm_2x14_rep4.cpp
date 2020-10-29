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

#include "convolution_kernel_mmad_slm_2x14_rep4.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey ConvolutionKernel_mmad_slm_2x14_rep4::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
    k.EnableOutputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.DisableTuning();
    return k;
}

bool ConvolutionKernel_mmad_slm_2x14_rep4::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o) || !CovolutionCheckInput(p, o)) {
        return false;
    }

    const convolution_params& cp = static_cast<const convolution_params&>(p);

    if (cp.filterSize.x != 3 || cp.filterSize.y != 3)
        return false;

    if (cp.inputs[0].X().v != 56 || cp.inputs[0].Y().v != 56)
        return false;

    if (cp.stride.x != 1 || cp.stride.y != 1)
        return false;

    return true;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_mmad_slm_2x14_rep4::SetDefault(const convolution_params& arg,
                                                                                     int) const {
    DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

    runInfo.efficiency = FORCE_PRIORITY_1;

    const size_t rep_count = 4;
    const size_t batch_per_wi = 1;
    const size_t out_block_width = 14;
    const size_t out_block_height = 2;
    runInfo.gws0 = arg.output.Feature().v *
                   (arg.output.Batch().v / (rep_count * batch_per_wi));  // number of tiles needed to cover output width
    runInfo.gws1 = ((arg.inputs[0].X().v / arg.stride.x) + (out_block_width - 1)) / out_block_width;
    runInfo.gws2 = ((arg.inputs[0].Y().v / arg.stride.y) + (out_block_height - 1)) / out_block_height;

    runInfo.lws0 = 32;  // depth
    runInfo.lws1 = 1;   // width
    runInfo.lws2 = 4;   // height

    return runInfo;
}

JitConstants ConvolutionKernel_mmad_slm_2x14_rep4::GetJitConstants(const convolution_params& params,
                                                                   const DispatchData& runInfo) const {
    auto jit = ConvolutionKernelBase::GetJitConstants(params, runInfo);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", 8));

    // pitch for special block format used in this kernel
    const size_t ifm_32_aligned = Align(params.weights.IFM().v, 32);
    const size_t filter_ofm_block_pitch =
        (ifm_32_aligned / 32) * params.weights.X().v * params.weights.Y().v * 4 * 8 * 8;
    jit.AddConstant(MakeJitConstant("FILTER_OFM_BLOCK_PITCH", filter_ofm_block_pitch));

    const size_t in_x_pitch = 32 * 4;
    const size_t in_y_pitch = 32 * 4 * params.inputs[0].X().LogicalDimPadded();
    const size_t in_b_block_pitch = in_y_pitch * params.inputs[0].Y().LogicalDimPadded();
    const size_t in_f_block_pitch = in_b_block_pitch * ((params.inputs[0].Batch().v + 3) / 4);
    const size_t in_offset =
        in_x_pitch * params.inputs[0].X().pad.before + in_y_pitch * params.inputs[0].Y().pad.before;

    jit.AddConstant(MakeJitConstant("IN_X_PITCH", in_x_pitch));
    jit.AddConstant(MakeJitConstant("IN_Y_PITCH", in_y_pitch));
    jit.AddConstant(MakeJitConstant("IN_B_BLOCK_PITCH", in_b_block_pitch));
    jit.AddConstant(MakeJitConstant("IN_F_BLOCK_PITCH", in_f_block_pitch));
    jit.AddConstant(MakeJitConstant("IN_OFFSET", in_offset));

    jit.AddConstant(MakeJitConstant("OUT_BLOCK_WIDTH", 14));
    jit.AddConstant(MakeJitConstant("OUT_BLOCK_HEIGHT", 2));
    jit.AddConstant(MakeJitConstant("LOCAL_SIZE_X", runInfo.lws0));
    jit.AddConstant(MakeJitConstant("LOCAL_SIZE_Y", runInfo.lws1));
    jit.AddConstant(MakeJitConstant("LOCAL_SIZE_Z", runInfo.lws2));

    return jit;
}

KernelsData ConvolutionKernel_mmad_slm_2x14_rep4::GetKernelsData(const Params& params,
                                                                 const optional_params& options) const {
    return GetCommonKernelsData(params, options, " -Dcl_intel_subgroups_char");
}
}  // namespace kernel_selector
