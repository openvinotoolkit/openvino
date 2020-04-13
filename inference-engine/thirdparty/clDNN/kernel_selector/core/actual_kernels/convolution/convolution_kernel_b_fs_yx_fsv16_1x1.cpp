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


#include <iostream>
#include "convolution_kernel_b_fs_yx_fsv16_1x1.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
static const size_t sub_group_size = 16;
static const size_t feature_block_size = 16;

ConvolutionKernel_b_fs_yx_fsv16_1x1::ConvolutionKernel_b_fs_yx_fsv16_1x1() : ConvolutionKernelBase("convolution_gpu_bfyx_f16_1x1") {
    std::vector<size_t> outputBlockWidths = {2, 4, 8};
    std::vector<std::string> executionModes = ConvolutionKernelBase::autoTuneOptions;

    for (auto w : outputBlockWidths) {
        for (auto exeMode : executionModes) {
            autoTuneOptions.emplace_back(AutoTuneOption{w, exeMode});
        }
    }
}

ConvolutionKernel_b_fs_yx_fsv16_1x1::AutoTuneOption ConvolutionKernel_b_fs_yx_fsv16_1x1::GetAutoTuneOptions(const Params& params,
                                                                                          int /*autoTuneIndex*/) const {
    const convolution_params& cp = static_cast<const convolution_params&>(params);
    auto x = cp.output.X().v;
    auto f = cp.output.Feature().v;
    if (x * f <= 256) {
        if ( x < 8 || x * f <= 128)
            return { 2, DEFAULT };
        else
            return { 4, DEFAULT };
    } else if (x * f <= 1536) {
        return { 4, DEFAULT };
    } else {
        return { 8, DEFAULT };
    }
}


ParamsKey ConvolutionKernel_b_fs_yx_fsv16_1x1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_b_fs_yx_fsv16_1x1::SetDefault(const convolution_params& params,
                                                                               int autoTuneIndex) const {
    DispatchData kd = ConvolutionKernelBase::SetDefault(params);

    auto autoTune = GetAutoTuneOptions(params, autoTuneIndex);
    kd.cldnnStyle.blockWidth = autoTune.blockWidth;

    const auto& out = params.output;
    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    kd.gws0 = CeilDiv(x * y, autoTune.blockWidth);
    kd.gws1 = Align(f, feature_block_size);
    kd.gws2 = b;

    kd.lws0 = 1;
    kd.lws1 = sub_group_size;
    kd.lws2 = 1;

    if (b == 1) {
        if (x <= 8)
            kd.efficiency = FORCE_PRIORITY_1;
        else
            kd.efficiency = FORCE_PRIORITY_2;
    } else {
        kd.efficiency = FORCE_PRIORITY_7;
    }

    return kd;
}

bool ConvolutionKernel_b_fs_yx_fsv16_1x1::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    const bool bOutputSizes =
        output.X().v != input.X().v || output.Y().v != input.Y().v || output.Feature().v % 16 != 0;
    const bool bFilterSize = params.filterSize.x != 1 || params.filterSize.y != 1;
    const bool bStride = params.stride.x != 1 || params.stride.y != 1;

    if  (bOutputSizes || bFilterSize || bStride) {
        return false;
    }

    return true;
}

JitConstants ConvolutionKernel_b_fs_yx_fsv16_1x1::GetJitConstants(const convolution_params& params,
                                                             const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    auto blockWidth = runInfo.cldnnStyle.blockWidth;
    if (!params.fused_ops.empty()) {
        auto input_dt = GetUnitType(params);
        FusedOpsConfiguration conf_vec = { "_VEC",
                                           {"b", "(f_block*16)", "y", "x"},
                                           "dst",
                                           input_dt,
                                           blockWidth,
                                           LoadType::LT_ALIGNED_READ,
                                           BoundaryCheck::ENABLED,
                                           IndexType::TENSOR_COORD,
                                           Tensor::DataChannelName::X };
        FusedOpsConfiguration conf_scalar = { "_SCALAR",
                                              {"b", "(f_block*16)", "yi", "xi"},
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
    jit.AddConstant(MakeJitConstant("PADDED_INPUT", params.inputs[0].X().pad.Total() != 0));

    bool padded_output = params.output.X().pad.Total() != 0;

    // Set padded_output to true when fused inputs have paddings to have correct blocked loads
    for (auto& fused_op : params.fused_ops) {
        for (auto& t : fused_op.tensors) {
            if (t.PitchesDifferFromLogicalDims()) {
                padded_output = true;
            }
        }
    }

    jit.AddConstant(MakeJitConstant("PADDED_OUTPUT", padded_output));

    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", blockWidth));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.output.X().v, blockWidth)));
    jit.AddConstant(MakeJitConstant("IC_BLOCKS", CeilDiv(params.inputs[0].Feature().v, feature_block_size)));
    if (params.output.Feature().v % feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
    }
    if (params.inputs[0].Feature().v % feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("INPUT_LEFTOVERS", 1));
    }

    return jit;
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16_1x1::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, DEFAULT, -1);
}

}  // namespace kernel_selector
