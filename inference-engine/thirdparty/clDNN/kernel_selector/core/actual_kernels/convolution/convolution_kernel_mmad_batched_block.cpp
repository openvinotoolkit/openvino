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

#include "convolution_kernel_mmad_batched_block.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {

ParamsKey ConvolutionKernel_mmad_batched_block::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
    k.EnableOutputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableBatching();
    k.EnableInt8Quantization();
    k.EnableOutputCalibration();
    k.DisableTuning();
    return k;
}

struct block_params {
    int32_t out_width;
    int32_t out_height;
    int32_t out_depth;
};

static block_params get_out_block_size(const convolution_params& p) {
    if (p.filterSize.x == 3 && p.filterSize.y == 3) {
        if (p.output.X().v == 7)
            return {7, 1, 4};
        else if (p.output.X().v == 14)
            return {7, 1, 4};
        else if (p.output.X().v == 28)
            return {7, 1, 4};
        else if (p.output.X().v == 56)
            return {8, 1, 4};
    }

    return {1, 1, 1};
}

std::vector<WeightsLayout> ConvolutionKernel_mmad_batched_block::GetSupportedWeightLayouts(
    const convolution_params& cp) const {
    auto block = get_out_block_size(cp);
    if (block.out_depth == 4)
        return {WeightsLayout::os_is_yx_isa8_osv8_isv4_swizzled_by_4};
    else
        return {WeightsLayout::os_is_yx_isa8_osv8_isv4};
}

bool ConvolutionKernel_mmad_batched_block::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o) || !CovolutionCheckInput(p, o)) {
        return false;
    }
    const convolution_params& cp = static_cast<const convolution_params&>(p);

    // if block sizes are 1x1, then this algorithm is probably not the best
    auto block = get_out_block_size(cp);
    if (block.out_width == 1 && block.out_height == 1)
        return false;

    if (cp.output.X().v % block.out_width != 0)
        return false;
    if (cp.output.Y().v % block.out_height != 0)
        return false;

    if (cp.filterSize.x == 1)
        return false;

    return true;
}

size_t static get_wg_batch_count(const convolution_params& params) {
    if (params.inputs[0].Batch().v % 64 == 0)
        return 16;  // because we process 4 batches per SIMD
    return 1;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_mmad_batched_block::SetDefault(const convolution_params& arg,
                                                                                     int) const {
    DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

    constexpr size_t sub_group_size = 8;

    runInfo.effiency = FORCE_PRIORITY_5;

    auto block = get_out_block_size(arg);

    runInfo.gws0 = arg.output.X().v / block.out_width;
    runInfo.gws1 = arg.output.Y().v / block.out_height;
    runInfo.gws2 = (arg.output.Feature().v) * ((arg.output.Batch().v + 3) / 4) /
                   block.out_depth;  // process 4 output channels per Workitem

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = sub_group_size * get_wg_batch_count(arg);

    return runInfo;
}

JitConstants ConvolutionKernel_mmad_batched_block::GetJitConstants(const convolution_params& params,
                                                                   const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    const int sub_group_size = 8;
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));

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

    const size_t out_x_pitch = 32 * 4;
    jit.AddConstant(MakeJitConstant("OUT_X_PITCH", out_x_pitch));

    auto block = get_out_block_size(params);
    jit.AddConstant(MakeJitConstant("OUT_BLOCK_WIDTH", block.out_width));
    jit.AddConstant(MakeJitConstant("OUT_BLOCK_HEIGHT", block.out_height));
    jit.AddConstant(MakeJitConstant("WEIGHTS_PER_WORKITEM", block.out_depth));

    jit.AddConstant(MakeJitConstant("WG_BATCH_COUNT", get_wg_batch_count(params)));

    return jit;
}

KernelsData ConvolutionKernel_mmad_batched_block::GetKernelsData(const Params& params,
                                                                 const optional_params& options) const {
    KernelsData kd = GetCommonKernelsData(params, options);
    if (!kd.empty())
        kd[0].estimatedTime = FORCE_PRIORITY_5;
    return kd;
}
}  // namespace kernel_selector