/*
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
*/

#include "fused_conv_eltwise_kernel_bfyx_1x1_opt.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ParamsKey fused_conv_eltwise_kernel_bfyx_1x1_opt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableSubGroup();
    // k.EnableSubGroupShort(); // we need it for FP16 only. we check it on the Validate phase
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableFusedConvEltwSplitSupport();
    k.EnableFusedConvEltwiseRWOutOpt();  // data for second input are already in output
    return k;
}

struct block_params {
    int32_t out_width;
    int32_t out_height;
    int32_t out_depth;
};

static block_params get_out_block_size(const fused_conv_eltwise_params& p) {
    auto out_depth = 8;

    if (p.output.X().v == 7) {
        auto gws0 = p.output.X().v / 7;
        auto gws1 = p.output.Y().v / 1;
        auto gws2 = 2 * (p.output.Feature().v * p.output.Batch().v) / 8;  // process 8 output channels per Workitem

        auto compute_units = p.engineInfo.computeUnitsCount;
        auto total_threads = (gws0 * gws1 * gws2) / 64;
        if (total_threads < compute_units) {
            out_depth /= 2;
            total_threads *= 2;
        }
        if (total_threads < compute_units) {
            out_depth /= 2;
            total_threads *= 2;
        }
        return {7, 1, out_depth};
    } else if (p.output.X().v == 14) {
        return {7, 1, 8};
    } else if (p.output.X().v == 28) {
        return {7, 2, 4};
    } else if (p.output.X().v == 56) {
        return {8, 1, 8};
    }

    return {1, 1, 1};
}

std::string fused_conv_eltwise_kernel_bfyx_1x1_opt::GetKernelName(const fused_conv_eltwise_params& params) const {
    if (params.inputs[0].GetDType() == Datatype::F32) {
        return kernelName + "_fp32";
    } else {
        return kernelName + "_fp16";
    }
}

bool fused_conv_eltwise_kernel_bfyx_1x1_opt::Validate(const Params& p, const optional_params& o) const {
    if (!fused_conv_eltwise_kernel_base::Validate(p, o) || !FusedConvolutionEltwiseCheckInput(p, o)) {
        return false;
    }

    const fused_conv_eltwise_params& cp = static_cast<const fused_conv_eltwise_params&>(p);

    if (cp.conv.stride.x != 1 || cp.conv.stride.y != 1)
        return false;

    if (cp.conv.filterSize.x != 1 || cp.conv.filterSize.y != 1)
        return false;

    if (cp.output.Feature().v % 64 != 0)
        return false;

    if (cp.conv.padding.x != 0 || cp.conv.padding.y != 0)
        return false;

    // if block sizes are 1x1, then this algorithm is probably not the best
    auto block = get_out_block_size(cp);
    if (block.out_width == 1 && block.out_height == 1)
        return false;

    if (cp.output.X().v % block.out_width != 0)
        return false;
    if (cp.output.Y().v % block.out_height != 0)
        return false;

    return true;
}

WeightsLayout fused_conv_eltwise_kernel_bfyx_1x1_opt::GetPreferreddWeightsLayout(
    const fused_conv_eltwise_params& p) const {
    auto block = get_out_block_size(p);
    if (block.out_depth == 8)
        return WeightsLayout::os_iyx_osv64;
    if (block.out_depth == 4)
        return WeightsLayout::os_iyx_osv32;
    if (block.out_depth == 2)
        return WeightsLayout::os_iyx_osv16;
    else
        return WeightsLayout::yxio;
}

fused_conv_eltwise_kernel_base::DispatchData fused_conv_eltwise_kernel_bfyx_1x1_opt::SetDefault(
    const fused_conv_eltwise_params& arg,
    int) const {
    DispatchData dispatchData = Parent::SetDefault(arg);

    constexpr size_t sub_group_size = 8;

    auto block = get_out_block_size(arg);

    dispatchData.gws[0] = arg.output.X().v / block.out_width;
    dispatchData.gws[1] = arg.output.Y().v / block.out_height;
    dispatchData.gws[2] = 2 * (arg.output.Feature().v * arg.output.Batch().v) /
                          block.out_depth;  // process 8 output channels per Workitem

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 2 * sub_group_size;

    return dispatchData;
}

KernelsPriority fused_conv_eltwise_kernel_bfyx_1x1_opt::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_1;
}

JitConstants fused_conv_eltwise_kernel_bfyx_1x1_opt::GetJitConstants(const fused_conv_eltwise_params& params,
                                                                     const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    auto block = get_out_block_size(params);
    jit.AddConstant(MakeJitConstant("OUT_BLOCK_WIDTH", block.out_width));
    jit.AddConstant(MakeJitConstant("OUT_BLOCK_HEIGHT", block.out_height));
    jit.AddConstant(MakeJitConstant("OUT_BLOCK_DEPTH", block.out_depth));

    return jit;
}

KernelsData fused_conv_eltwise_kernel_bfyx_1x1_opt::GetKernelsData(const Params& params,
                                                                   const optional_params& options) const {
    KernelsData kd = GetCommonKernelsData(params, options);
    return kd;
}
}  // namespace kernel_selector
