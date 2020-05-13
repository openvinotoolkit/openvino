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


#include <iostream>
#include "convolution_kernel_b_fs_yx_fsv16_depthwise.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
static const size_t sub_group_size = 16;
static const size_t feature_block_size = 16;
static const size_t x_block_size = 8;

ParamsKey ConvolutionKernel_b_fs_yx_fsv16_depthwise::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableGroupedConvolution();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    k.EnableDepthwiseSeparableOpt();
    k.EnableDilation();
    return k;
}

bool ConvolutionKernel_b_fs_yx_fsv16_depthwise::Validate(const Params& p, const optional_params&) const {
    const convolution_params& cp = static_cast<const convolution_params&>(p);

    if (cp.groups == 1)
        return false;

    if (cp.inputs[0].Feature().v != cp.groups || cp.output.Feature().v != cp.groups)
        return false;

    // Check that padding features doesn't miss-align the blocks
    if (cp.inputs[0].Feature().pad.before % feature_block_size != 0 || cp.output.Feature().pad.before % feature_block_size != 0)
        return false;

    return true;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_b_fs_yx_fsv16_depthwise::SetDefault(const convolution_params& params,
                                                                                     int) const {
    DispatchData runInfo = Parent::SetDefault(params);
    const auto& out = params.output;

    runInfo.gws0 = CeilDiv(out.X().v, x_block_size) * out.Y().v;
    runInfo.gws1 = Align(out.Feature().v, feature_block_size);
    runInfo.gws2 = out.Batch().v;
    runInfo.lws0 = 1;
    runInfo.lws1 = sub_group_size;
    runInfo.lws2 = 1;

    if (out.Batch().v == 1)
        runInfo.efficiency = FORCE_PRIORITY_1;
    else
        runInfo.efficiency = FORCE_PRIORITY_7;

    return runInfo;
}

JitConstants ConvolutionKernel_b_fs_yx_fsv16_depthwise::GetJitConstants(const convolution_params& params,
                                                                   const DispatchData& kd) const {
    auto jit = ConvolutionKernelBase::GetJitConstants(params, kd);

    const size_t block_width = 8;

    if (!params.fused_ops.empty()) {
        auto input_dt = GetUnitType(params);
        FusedOpsConfiguration conf_vec = { "_VEC", {"b", "(f_block*16)", "y", "x"},
                                           "dst",
                                           input_dt,
                                           block_width,
                                           LoadType::LT_ALIGNED_READ,
                                           BoundaryCheck::ENABLED,
                                           IndexType::TENSOR_COORD,
                                           Tensor::DataChannelName::X };
        FusedOpsConfiguration conf_scalar = { "_SCALAR",
                                              {"b", "(f_block*16)", "y", "(x+i)"},
                                              "dst[i]",
                                              input_dt,
                                              1,
                                              LoadType::LT_ALIGNED_READ,
                                              BoundaryCheck::ENABLED,
                                              IndexType::TENSOR_COORD,
                                              Tensor::DataChannelName::X };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_vec, conf_scalar}));
    }

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.output.X().v, block_width)));
    jit.AddConstant(MakeJitConstant("IC_BLOCK", feature_block_size));
    jit.AddConstant(MakeJitConstant("FILTER_SIZE_X_DIV_2", params.filterSize.x / 2));
    if (params.output.Feature().v % feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
    }

    return jit;
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16_depthwise::GetKernelsData(const Params& params,
                                                                 const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

}  // namespace kernel_selector
