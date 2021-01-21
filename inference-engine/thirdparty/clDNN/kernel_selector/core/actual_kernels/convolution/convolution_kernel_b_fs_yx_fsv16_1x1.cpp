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

ConvolutionKernel_b_fs_yx_fsv16_1x1::ConvolutionKernel_b_fs_yx_fsv16_1x1() : ConvolutionKernelBase("convolution_gpu_bfyx_f16_1x1") {
    std::vector<size_t> outputBlockWidths = { 1, 2, 4, 8 };
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
    auto y = cp.output.Y().v;
    auto f = cp.output.Feature().v;

    if (x == 1 && y == 1) {
        return { 1, DEFAULT };
    } else if (x * f <= 256) {
        if (x < 8 || x * f <= 128)
            return { 2, DEFAULT };
        else
            return { 4, DEFAULT };
    } else if (x * f <= 1536) {
        return { 4, DEFAULT };
    } else {
        return { 8, DEFAULT };
    }
}

float ConvolutionKernel_b_fs_yx_fsv16_1x1::EstimateOccupancy(const convolution_params& params,
                                                             const ConvolutionTuningData& tuning_data) const {
    auto tuneOptions = GetAutoTuneOptions(params, 0);
    auto blockWidth = tuneOptions.blockWidth;

    auto x = params.output.X().v;
    auto y = params.output.Y().v;
    auto f = params.output.Feature().v;
    auto b = params.output.Batch().v;

    auto threads = CeilDiv(x * y, blockWidth) * CeilDiv(f, tuning_data.feature_block_size) * tuning_data.slm_div_factor * b;

    return static_cast<float>(threads) / static_cast<float>(params.engineInfo.maxThreadsPerDevice);
}

ConvolutionKernel_b_fs_yx_fsv16_1x1::ConvolutionTuningData ConvolutionKernel_b_fs_yx_fsv16_1x1::GetTuningParams(const convolution_params& params) const {
    ConvolutionTuningData tuning_data;

    const auto& input = params.inputs[0];

    size_t ic_blocks = CeilDiv(input.Feature().v, tuning_data.feature_block_size);

    size_t max_slm_div_factor = params.engineInfo.maxWorkGroupSize / tuning_data.sub_group_size;
    bool block_size_one_is_better = params.output.X().v == 1 && params.output.Y().v == 1 && input.Feature().v >= 2048;

    if (params.engineInfo.deviceType == dev_type::integrated_gpu && params.engineInfo.bIMADSupport && !block_size_one_is_better)
        while (ic_blocks % (tuning_data.slm_div_factor * 2) == 0 && (tuning_data.slm_div_factor * 2 <= max_slm_div_factor) &&
               EstimateOccupancy(params, tuning_data) < 4.0)
            tuning_data.slm_div_factor *= 2;

    tuning_data.work_group_size = tuning_data.slm_div_factor * tuning_data.sub_group_size;

    return tuning_data;
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
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(params);

    ConvolutionTuningData tuning_data = GetTuningParams(params);

    auto autoTune = GetAutoTuneOptions(params, autoTuneIndex);
    dispatchData.cldnnStyle.blockWidth = autoTune.blockWidth;

    const auto& out = params.output;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    dispatchData.gws[0] = x == 1 && y == 1 ? 1 : CeilDiv(x * y, autoTune.blockWidth);
    dispatchData.gws[1] = Align(f, tuning_data.feature_block_size) * tuning_data.slm_div_factor;
    dispatchData.gws[2] = b;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = tuning_data.work_group_size;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_b_fs_yx_fsv16_1x1::GetKernelsPriority(const Params& params, const optional_params& /*options*/) const {
    const auto& p = static_cast<const convolution_params&>(params);
    auto autoTune = GetAutoTuneOptions(params, -1);

    const auto& input = p.inputs[0];
    const auto& out = p.output;

    auto bBlockSizeX = out.X().v % autoTune.blockWidth == 0;
    auto bBlockSizeXY = out.X().pad.Total() + out.Y().pad.Total() == 0;
    auto bInputPad = input.X().pad.Total() + input.Y().pad.Total() != 0;

    if (out.Batch().v == 1) {
        if ((bBlockSizeX || bBlockSizeXY) && !bInputPad) {
            return FORCE_PRIORITY_1;
        } else {
            return FORCE_PRIORITY_3;
        }
    } else {
        return FORCE_PRIORITY_7;
    }
}

bool ConvolutionKernel_b_fs_yx_fsv16_1x1::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    ConvolutionTuningData tuning_data = GetTuningParams(params);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    const bool bOutputSizes = output.X().v != input.X().v || output.Y().v != input.Y().v || output.Feature().v % 16 != 0;
    const bool bFilterSize = params.filterSize.x != 1 || params.filterSize.y != 1;
    const bool bStride = params.stride.x != 1 || params.stride.y != 1;
    const bool bPadding = input.Feature().pad.before % tuning_data.feature_block_size != 0 ||
                          output.Feature().pad.before % tuning_data.feature_block_size != 0;

    if  (bOutputSizes || bFilterSize || bStride || bPadding) {
        return false;
    }

    return true;
}

JitConstants ConvolutionKernel_b_fs_yx_fsv16_1x1::GetJitConstants(const convolution_params& params,
                                                                  const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    ConvolutionTuningData tuning_data = GetTuningParams(params);

    auto blockWidth = dispatchData.cldnnStyle.blockWidth;
    if (!params.fused_ops.empty()) {
        auto input_dt = GetUnitType(params);
        FusedOpsConfiguration conf_vec = { "_VEC",
                                           {"b", "(feature_block * 16)", "y", "x"},
                                           "dst",
                                           input_dt,
                                           blockWidth,
                                           LoadType::LT_ALIGNED_READ,
                                           BoundaryCheck::ENABLED,
                                           IndexType::TENSOR_COORD,
                                           Tensor::DataChannelName::X };
        FusedOpsConfiguration conf_scalar1 = { "_SCALAR",
                                              {"b", "(feature_block * 16)", "yi", "xi"},
                                              "dst[i]",
                                              input_dt,
                                              1,
                                              LoadType::LT_ALIGNED_READ,
                                              BoundaryCheck::ENABLED,
                                              IndexType::TENSOR_COORD,
                                              Tensor::DataChannelName::X };
        FusedOpsConfiguration conf_scalar2 = { "_SCALAR_B1",
                                              {"b", "(feature_block * 16)", "0", "0"},
                                              "dst",
                                              input_dt,
                                              1,
                                              LoadType::LT_ALIGNED_READ,
                                              BoundaryCheck::ENABLED,
                                              IndexType::TENSOR_COORD,
                                              Tensor::DataChannelName::X };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf_vec, conf_scalar1, conf_scalar2 }));
    }

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", tuning_data.sub_group_size));
    jit.AddConstant(MakeJitConstant("PADDED_INPUT", params.inputs[0].X().pad.Total() != 0));

    bool padded_output = params.output.X().pad.Total() != 0;
    bool non_unit_fused_op_spatial = false;

    // Set padded_output to true when fused inputs have paddings to have correct blocked loads
    for (auto& fused_op : params.fused_ops) {
        for (auto& t : fused_op.tensors) {
            if (t.PitchesDifferFromLogicalDims()) {
                padded_output = true;
            }
            if ((t.X().v > 1) ||
                (t.Y().v > 1) ||
                (t.Z().v > 1) ||
                (t.W().v > 1)) {
                non_unit_fused_op_spatial = true;
            }
        }
    }

    jit.AddConstant(MakeJitConstant("PADDED_OUTPUT", padded_output));
    jit.AddConstant(MakeJitConstant("NON_UNIT_FUSED_OP_SPATIAL", non_unit_fused_op_spatial));

    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", blockWidth));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.output.X().v, blockWidth)));
    jit.AddConstant(MakeJitConstant("SLM_DIV_FACTOR", tuning_data.slm_div_factor));
    jit.AddConstant(MakeJitConstant("WORK_GROUP_SIZE", tuning_data.work_group_size));
    jit.AddConstant(MakeJitConstant("IC_BLOCKS", CeilDiv(params.inputs[0].Feature().v, tuning_data.feature_block_size)));
    if (params.output.Feature().v % tuning_data.feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
    }
    if (params.inputs[0].Feature().v % tuning_data.feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("INPUT_LEFTOVERS", 1));
    }

    return jit;
}

KernelsData ConvolutionKernel_b_fs_yx_fsv16_1x1::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, DEFAULT, -1);
}

}  // namespace kernel_selector
