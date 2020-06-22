// Copyright (c) 2019-2020 Intel Corporation
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

#include "convolution_kernel_mmad_bfyx_to_b_fs_yx_fsv32.h"
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include <iostream>

namespace kernel_selector {

static const size_t sub_group_size = 16;

ParamsKey ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_WEIGHTS);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS);
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.DisableTuning();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

bool ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::Validate(const Params &p, const optional_params &o) const {
    if (!Parent::Validate(p, o)) {
        return false;
    }

    auto params = dynamic_cast<const convolution_params&>(p);

    if (params.inputs[0].Feature().v != 3 && params.inputs[0].Feature().v != 4)
        return false;

    if (params.output.Feature().v % 2 != 0)
        return false;

    if ((params.quantization == QuantizationType::ASYMMETRIC_DATA || params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS)
        && !params.HasCompensation()) {
        return false;
    }

    return true;
}

ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::AutoTuneOption ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::GetAutoTuneOptions(const Params &p,
                                                                                                                        int autoTuneIndex) const {
    if ((autoTuneIndex >= 0) && (autoTuneIndex < static_cast<int>(autoTuneOptions.size()))) {
        return autoTuneOptions[autoTuneIndex];
    }

    AutoTuneOption option = {0, 0, 0, DEFAULT};

    auto &params = dynamic_cast<const convolution_params &>(p);
    auto &output = params.output;

    // TODO: Check if other block size can improve performance
    option.blockHeight = 1;
    option.prefetch = 1;
    auto estimateRegUsage = [&](size_t blockWidth, size_t simd) {
        static const size_t activation_type_size = 4;
        static const size_t output_features_per_wi = 2;
        size_t bytes_per_simd = 0;
        // Accumulation matrix
        bytes_per_simd += activation_type_size * output_features_per_wi * blockWidth;
        size_t input_line_size = std::min(params.stride.x * (blockWidth - 1) + (params.weights.X().v - 1) * params.dilation.x + 1,
                                          params.inputs[0].X().v + params.inputs[0].X().pad.Total());
        // Input line
        bytes_per_simd += activation_type_size * input_line_size;
        // Weights zero point accumulator
        if (!params.weights_zero_points.empty()) {
            bytes_per_simd += activation_type_size * blockWidth;
        }
        // Weights
        bytes_per_simd += activation_type_size * output_features_per_wi;
        // Extra variables: input_x, input_y, input, filter offset, b, fg, y, x, lid
        bytes_per_simd += 4 * 9;
        return bytes_per_simd * simd;
    };
    static const size_t registers_count = 128;
    static const size_t register_byte_size = 32;
    static const size_t register_bytes = registers_count * register_byte_size;
    static const size_t max_register_bytes = register_bytes * 3 / 4;
    static const size_t simd_size = 16;
    if (output.LogicalSize() > 49 * 1024 && estimateRegUsage(8, simd_size) <= max_register_bytes) {
        option.blockWidth = 8;
    } else {
        option.blockWidth = 4;
    }

    return option;
}

static size_t get_slm_byte_size(const convolution_params &cp, size_t lws, size_t block_size) {
    return (cp.stride.x * (lws * block_size - 1) + (cp.weights.X().v - 1) * cp.dilation.x + 1)*
            cp.weights.Y().v * sizeof(int32_t);
}

static size_t get_lws(const convolution_params &cp, size_t blocks_count, size_t block_size, size_t max_lws) {
    while (max_lws > 1) {
        if (blocks_count % max_lws == 0) {
            if (get_slm_byte_size(cp, max_lws, block_size) < cp.engineInfo.maxLocalMemSize)
                return max_lws;
        }
        max_lws--;
    }

    return 1;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::SetDefault(const convolution_params &cp,
                                                                                             int autoTuneIndex) const {
    DispatchData runInfo = ConvolutionKernelBase::SetDefault(cp);

    auto tuneOptions = GetAutoTuneOptions(cp, autoTuneIndex);
    runInfo.cldnnStyle.blockWidth = tuneOptions.blockWidth;
    runInfo.cldnnStyle.blockHeight = tuneOptions.blockHeight;
    runInfo.cldnnStyle.prefetch = tuneOptions.prefetch;

    runInfo.efficiency = FORCE_PRIORITY_3;

    const size_t max_lws = std::max((size_t)1, cp.engineInfo.maxWorkGroupSize / sub_group_size);
    runInfo.gws0 = Align(cp.output.Feature().v, 32) / 2;
    runInfo.gws1 = CeilDiv(cp.output.X().v, runInfo.cldnnStyle.blockWidth);
    runInfo.gws2 = cp.output.Batch().v * cp.output.Y().v;

    runInfo.lws0 = sub_group_size;
    runInfo.lws1 = get_lws(cp, runInfo.gws1, tuneOptions.blockWidth, max_lws);
    runInfo.lws2 = 1;

    return runInfo;
}

JitConstants ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::GetJitConstants(const convolution_params &params,
                                                                           const DispatchData &runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", runInfo.lws0));
    jit.AddConstant(MakeJitConstant("LWS0", runInfo.lws0));
    jit.AddConstant(MakeJitConstant("LWS1", runInfo.lws1));
    jit.AddConstant(MakeJitConstant("LWS2", runInfo.lws2));
    jit.AddConstant(MakeJitConstant("OSV", 32));
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", runInfo.cldnnStyle.blockWidth));
    auto input = params.inputs[0];
    auto output = params.output;
    auto blockWidth = runInfo.cldnnStyle.blockWidth;
    size_t slm_line_size = params.stride.x * (runInfo.lws1 * blockWidth - 1) + (params.weights.X().v - 1) * params.dilation.x + 1;
    size_t slm_chunk_size = slm_line_size / runInfo.lws1;
    size_t slm_tail = slm_line_size % runInfo.lws1;
    size_t slm_line_aligned = slm_chunk_size*runInfo.lws1 + Align(slm_tail, sub_group_size);

    size_t input_line_size = std::min(params.stride.x * (blockWidth - 1) + (params.weights.X().v - 1) * params.dilation.x + 1,
                                      input.X().v + input.X().pad.Total());

    jit.AddConstant(MakeJitConstant("INPUT_LINE_SIZE", input_line_size));
    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", blockWidth));
    jit.AddConstant(MakeJitConstant("GROUP_SIZE", blockWidth * runInfo.lws1));
    jit.AddConstant(MakeJitConstant("SLM_LINE_SIZE", slm_line_aligned));
    jit.AddConstant(MakeJitConstant("SLM_CHUNK_SIZE", slm_chunk_size));
    jit.AddConstant(MakeJitConstant("SLM_TAIL", slm_tail));

    jit.Merge(MakeTypeJitConstants(GetPackedInputType(params), "PACKED_IN"));
    jit.Merge(MakeTypeJitConstants(GetPackedType(params.output.GetDType(), 2), "PACKED_OUT"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        if (GetPreferredWeightsLayout(params) == WeightsLayout::os_is_yx_osv32_isv4) {
            FusedOpsConfiguration conf0 = {"_0", {"b", "(fg*32 + lid)", "y", "(x+i)"}, "res0", input_dt, 1};
            FusedOpsConfiguration conf1 = {"_1", {"b", "(fg*32 + lid+16)", "y", "(x+i)"}, "res1", input_dt, 1};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf0, conf1}));
        } else {
            FusedOpsConfiguration conf0 = {"_0", {"b", "(fg*32 + 2*lid + 0)", "y", "(x+i)"}, "res0", input_dt, 1};
            FusedOpsConfiguration conf1 = {"_1", {"b", "(fg*32 + 2*lid + 1)", "y", "(x+i)"}, "res1", input_dt, 1};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf0, conf1}));
        }
    }

    return jit;
}

KernelsData ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::GetKernelsData(const Params &params, const optional_params &options) const {
    KernelsData kd = GetTunedKernelsDataByIndex(params, options);
    if (!kd.empty()) {
        kd[0].estimatedTime = FORCE_PRIORITY_2;
    }

    return kd;
}

KernelsData ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::GetKernelsDataForAutoTune(const Params &params,
                                                                                    const optional_params &options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelsData res = {};

    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params, options, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

}  // namespace kernel_selector
