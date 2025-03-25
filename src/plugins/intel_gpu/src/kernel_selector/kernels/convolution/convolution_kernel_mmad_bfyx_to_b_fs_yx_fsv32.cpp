// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
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
    return k;
}

DeviceFeaturesKey ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_blocked_read_write();
    k.requires_blocked_read_write_short();
    k.requires_blocked_read_write_char();
    k.requires_subgroups();

    return k;
}

bool ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::Validate(const Params &p) const {
    if (!Parent::Validate(p)) {
        return false;
    }

    auto params = dynamic_cast<const convolution_params&>(p);

    if (params.inputs[0].Dimentions() != params.outputs[0].Dimentions())
        return false;

    if (params.inputs[0].Feature().v != 3 && params.inputs[0].Feature().v != 4)
        return false;

    if (params.outputs[0].Feature().v % 2 != 0)
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

    AutoTuneOption option = {0, 0, 0, EXE_MODE_DEFAULT};

    auto &params = dynamic_cast<const convolution_params &>(p);
    auto &output = params.outputs[0];

    // TODO: Check if other block size can improve performance
    option.blockHeight = 1;
    option.prefetch = 1;
    auto estimateRegUsage = [&](size_t blockWidth, size_t blockHeight, size_t simd) {
        const size_t output_features_per_wi = 2 * blockHeight;
        static const size_t activation_type_size = 4;

        size_t bytes_per_simd = 0;
        // Accumulation matrix
        bytes_per_simd += activation_type_size * output_features_per_wi * blockWidth * blockHeight;
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
    if (output.LogicalSize() > 49 * 1024 && estimateRegUsage(8, 1, simd_size) <= max_register_bytes) {
        option.blockWidth = 8;
    } else if (estimateRegUsage(4, 2, simd_size) <= max_register_bytes && params.dilation.y == 1) {
        option.blockWidth = 4;
        option.blockHeight = 2;
    } else {
        option.blockWidth = 4;
    }

    return option;
}

static size_t get_slm_byte_size(const convolution_params &cp, size_t lws, size_t block_size_x, size_t block_size_y) {
    const size_t input_y_height = cp.stride.y * (block_size_y - 1) + (cp.weights.Y().v - 1) * cp.dilation.y + 1;
    return (cp.stride.x * (lws * block_size_x - 1) + (cp.weights.X().v - 1) * cp.dilation.x + 1) *
            input_y_height * cp.weights.Z().v * sizeof(int32_t);
}

static size_t get_lws(const convolution_params &cp, size_t blocks_count, size_t block_size_x, size_t block_size_y, size_t max_lws) {
    while (max_lws > 1) {
        if (blocks_count % max_lws == 0) {
            if (get_slm_byte_size(cp, max_lws, block_size_x, block_size_y) < cp.engineInfo.maxLocalMemSize)
                return max_lws;
        }
        max_lws--;
    }

    return 1;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::SetDefault(const convolution_params &cp,
                                                                                             int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(cp);

    auto tuneOptions = GetAutoTuneOptions(cp, autoTuneIndex);
    dispatchData.cldnnStyle.blockWidth = tuneOptions.blockWidth;
    dispatchData.cldnnStyle.blockHeight = tuneOptions.blockHeight;
    dispatchData.cldnnStyle.prefetch = tuneOptions.prefetch;

    const size_t max_lws = std::max((size_t)1, cp.engineInfo.maxWorkGroupSize / sub_group_size);
    dispatchData.gws[0] = Align(cp.outputs[0].Feature().v, 32) / 2;
    dispatchData.gws[1] = CeilDiv(cp.outputs[0].X().v, dispatchData.cldnnStyle.blockWidth);
    dispatchData.gws[2] = cp.outputs[0].Batch().v * CeilDiv(cp.outputs[0].Y().v, dispatchData.cldnnStyle.blockHeight) * cp.outputs[0].Z().v;

    dispatchData.lws[0] = sub_group_size;
    dispatchData.lws[1] = get_lws(cp, dispatchData.gws[1], tuneOptions.blockWidth, tuneOptions.blockHeight, max_lws);
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

JitConstants ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::GetJitConstants(const convolution_params &params,
                                                                           const DispatchData &dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", dispatchData.lws[0]));
    jit.AddConstant(MakeJitConstant("LWS0", dispatchData.lws[0]));
    jit.AddConstant(MakeJitConstant("LWS1", dispatchData.lws[1]));
    jit.AddConstant(MakeJitConstant("LWS2", dispatchData.lws[2]));
    jit.AddConstant(MakeJitConstant("OSV", 32));
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", dispatchData.cldnnStyle.blockWidth));
    auto input = params.inputs[0];
    auto output = params.outputs[0];
    auto blockWidth = dispatchData.cldnnStyle.blockWidth;
    auto blockHeight = dispatchData.cldnnStyle.blockHeight;
    size_t slm_line_size = params.stride.x * (dispatchData.lws[1] * blockWidth - 1) + (params.weights.X().v - 1) * params.dilation.x + 1;
    size_t slm_chunk_size = slm_line_size / dispatchData.lws[1];
    size_t slm_tail = slm_line_size % dispatchData.lws[1];
    size_t slm_line_aligned = slm_chunk_size*dispatchData.lws[1] + Align(slm_tail, sub_group_size);

    size_t input_line_size = params.stride.x * (blockWidth - 1) + (params.weights.X().v - 1) * params.dilation.x + 1;
    size_t input_y_height = blockHeight != 1 ? params.stride.y * (blockHeight - 1) + (params.weights.Y().v - 1) * params.dilation.y + 1 :
                                               params.weights.Y().v;

    jit.AddConstant(MakeJitConstant("INPUT_LINE_SIZE", input_line_size));
    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", blockWidth));
    jit.AddConstant(MakeJitConstant("GROUP_SIZE", blockWidth * dispatchData.lws[1]));
    jit.AddConstant(MakeJitConstant("OUTPUT_Y_BLOCK_SIZE", blockHeight));
    jit.AddConstant(MakeJitConstant("SLM_LINE_SIZE", slm_line_aligned));
    jit.AddConstant(MakeJitConstant("SLM_CHUNK_SIZE", slm_chunk_size));
    jit.AddConstant(MakeJitConstant("SLM_TAIL", slm_tail));
    jit.AddConstant(MakeJitConstant("INPUT_Y_HEIGHT", input_y_height));

    jit.Merge(MakeTypeJitConstants(GetPackedInputType(params), "PACKED_IN"));
    jit.Merge(MakeTypeJitConstants(GetPackedType(params.outputs[0].GetDType(), 2), "PACKED_OUT"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        if (WeightsTensor::ChannelsCount(GetPreferredWeightsLayout(params)) == 5) {
            FusedOpsConfiguration conf0 = {"_0", {"b", "(fg*32 + lid)", "z", "(y+j)", "(x+i)"}, "res0", input_dt, 1};
            FusedOpsConfiguration conf1 = {"_1", {"b", "(fg*32 + lid + 16)", "z", "(y+j)", "(x+i)"}, "res1", input_dt, 1};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf0, conf1}));
        } else {
            if (GetPreferredWeightsLayout(params) == WeightsLayout::os_is_yx_osv32_isv4) {
                FusedOpsConfiguration conf0 = {"_0", {"b", "(fg*32 + lid)", "(y+j)", "(x+i)"}, "res0", input_dt, 1};
                FusedOpsConfiguration conf1 = {"_1", {"b", "(fg*32 + lid + 16)", "(y+j)", "(x+i)"}, "res1", input_dt, 1};
                jit.Merge(MakeFusedOpsJitConstants(params, {conf0, conf1}));
            } else {
                FusedOpsConfiguration conf0 = {"_0", {"b", "(fg*32 + 2*lid + 0)", "(y+j)", "(x+i)"}, "res0", input_dt, 1};
                FusedOpsConfiguration conf1 = {"_1", {"b", "(fg*32 + 2*lid + 1)", "(y+j)", "(x+i)"}, "res1", input_dt, 1};
                jit.Merge(MakeFusedOpsJitConstants(params, {conf0, conf1}));
            }
        }
    }

    return jit;
}

KernelsData ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::GetKernelsData(const Params &params) const {
    KernelsData kd = GetTunedKernelsDataByIndex(params);
    return kd;
}

KernelsData ConvolutionKernel_mmad_bfyx_to_b_fs_yx_fsv32::GetKernelsDataForAutoTune(const Params &params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelsData res = {};

    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

}  // namespace kernel_selector
