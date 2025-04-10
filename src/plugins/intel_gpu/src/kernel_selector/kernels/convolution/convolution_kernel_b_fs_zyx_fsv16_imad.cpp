// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_b_fs_zyx_fsv16_imad.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

//
// Kernel specific constants
//
static constexpr size_t fsv = 16;
static constexpr size_t simd = 16;

static size_t getOutBlock_X(const size_t output_size_x, const size_t stride_x, const size_t filter_size_x, const size_t dilation_x) {
    // Calculate number of variables needed to hold minimum input width.
    // Equation for input block width: (output_block - 1) * stride + (filter_size - 1) * dilation + 1
    // Result for one output_block gives minimum size of input width.
    size_t min_in_block_size = (filter_size_x - 1) * dilation_x + 1;
    // Input block is spread across sub-group, so ceil-divide by simd size.
    size_t min_in_block_simds = kernel_selector::CeilDiv(min_in_block_size, simd);

    size_t output_block_width = 0;
    size_t max_block_size = std::min((min_in_block_simds * simd - 1 - (filter_size_x - 1) * dilation_x) / stride_x + 1, output_size_x);

    if (output_size_x <= max_block_size)
        return output_size_x;

    for (size_t block = 4; block <= max_block_size; ++block) {
        if (output_size_x % block == 0)
            output_block_width = block;
    }
    if (output_block_width == 0 && output_size_x < max_block_size * 3) {
        size_t min_overhang = max_block_size;
        for (size_t block = 4; block <= max_block_size; ++block) {
            size_t overhang = block - output_size_x % block;
            if (overhang <= min_overhang) {
                min_overhang = overhang;
                output_block_width = block;
            }
        }
    }

    if (output_block_width == 0) {
        output_block_width = max_block_size;
    }
    return output_block_width;
}

namespace kernel_selector {

Convolution_kernel_b_fs_zyx_fsv16_imad::BlockParams
Convolution_kernel_b_fs_zyx_fsv16_imad::GetBlockParams(const convolution_params& params) const {
    size_t max_block_width = 1;
    if (!params.outputs[0].X().is_dynamic) {
        max_block_width = getOutBlock_X(params.outputs[0].X().v, params.stride.x, params.filterSize.x, params.dilation.x);
    }
    size_t max_in_block_width = (max_block_width - 1) * params.stride.x + (params.filterSize.x - 1) * params.dilation.x + 1;

    size_t block_width = max_block_width;
    if (max_block_width > 1) {
        for (size_t w = max_block_width; w >= CeilDiv(max_block_width, 2); w -= 1) {
            if (params.outputs[0].X().v % w == 0) {
                block_width = w;
                break;
            }
        }
    }

    size_t in_block_width = (block_width - 1) * params.stride.x + (params.filterSize.x - 1) * params.dilation.x + 1;
    size_t block_features = simd;
    size_t feature_slm_split = 1;
    size_t block_height = 1;
    size_t block_depth = 1;
    size_t in_block_height = 1;
    size_t in_block_depth = 1;

    // Estimate basic block params ratio
    auto test_block_params = BlockParams{ block_width, 1, 1, simd, in_block_width, 1, 1, 1 };

    // Use default block parameters for asymmetric weights quantization for devices with immad support due to unoptimized tuning
    if ((params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS || params.quantization == QuantizationType::ASYMMETRIC_WEIGHTS) &&
        params.engineInfo.supports_immad) {
        return test_block_params;
    }

    auto best_block_params_ratio = EstimateBlockParamsRatio(params, test_block_params);

    size_t max_slm_split = params.engineInfo.maxWorkGroupSize / simd;

    // TGLU exceptions related to SLM usage
    if (params.is_shape_agnostic) {
        max_slm_split = 2;
    } else if (params.engineInfo.deviceType == dev_type::integrated_gpu && params.engineInfo.computeUnitsCount == 96) {
        bool split_exception_1 = params.outputs[0].X().v == 3 && params.outputs[0].Y().v == 3 && params.outputs[0].Z().v == 1
                                 && params.outputs[0].Feature().v == 512;
        bool split_exception_2 = params.outputs[0].X().v == 5 && params.outputs[0].Y().v == 5 && params.outputs[0].Z().v == 1
                                 && params.outputs[0].Feature().v == 256;
        bool split_exception_3 = params.outputs[0].X().v == 9 && params.outputs[0].Y().v == 9 && params.outputs[0].Z().v == 1
                                 && params.outputs[0].Feature().v == 128;
        bool split_exception_4 = params.outputs[0].X().v == 18 && params.outputs[0].Y().v == 18 && params.outputs[0].Z().v == 1
                                 && params.outputs[0].Feature().v == 64;

        if (split_exception_1 || split_exception_2 || split_exception_3 || split_exception_4)
            max_slm_split = 2;
    }

    // Check ratio in cycle for all available block params
    for (size_t w = 0; w < 2; w++) {
        size_t temp_block_width = block_width;
        size_t temp_in_block_width = in_block_width;

        if (w == 1) {
            if (max_block_width > 1) {
                temp_block_width = max_block_width;
                temp_in_block_width = max_in_block_width;
            } else {
                break;
            }
        }

        size_t max_d = params.outputs[0].Z().is_dynamic ? 1 : 16;
        size_t max_h = params.outputs[0].Y().is_dynamic ? 1 : 16;

        for (size_t split = 1; split <= max_slm_split; split *= 2) {
            for (size_t temp_block_features = simd; temp_block_features <= simd * 2; temp_block_features += simd) {
                for (size_t d = 1; d < max_d; ++d) {
                    if (d != 1 && params.outputs[0].Z().v % d)
                        continue;
                    for (size_t h = 1; h < max_h; ++h) {
                        if (h != 1 && params.outputs[0].Y().v % h)
                            continue;

                        bool c_ifm_mul = CeilDiv(params.weights.IFM().v, fsv) % split == 0;
                        bool c_mul_f = temp_block_features == simd ? true : params.weights.OFM().v % temp_block_features == 0;

                        size_t temp_block_height = 1;
                        size_t temp_block_depth = 1;
                        size_t temp_in_block_height = 1;
                        size_t temp_in_block_depth = 1;

                        if (h != 1) {
                            temp_block_height = h;
                            temp_block_depth = d;
                            temp_in_block_height = (h - 1) * params.stride.y + (params.filterSize.y - 1) * params.dilation.y + 1;
                            temp_in_block_depth = (d - 1) * params.stride.z + (params.filterSize.z - 1) * params.dilation.z + 1;
                        }

                        // Estimate current block params ratio
                        test_block_params = BlockParams{ temp_block_width, temp_block_height, temp_block_depth, temp_block_features,
                                                         temp_in_block_width, temp_in_block_height, temp_in_block_depth, split };
                        auto block_params_ratio = EstimateBlockParamsRatio(params, test_block_params);

                        // Try to increase block_params_ratio
                        if (c_ifm_mul && c_mul_f && block_params_ratio > best_block_params_ratio) {
                            best_block_params_ratio = block_params_ratio;

                            // Update block params if current ratio is better than previous
                            block_width = temp_block_width;
                            block_height = temp_block_height;
                            block_depth = temp_block_depth;
                            block_features = temp_block_features;

                            in_block_width = temp_in_block_width;
                            in_block_height = temp_in_block_height;
                            in_block_depth = temp_in_block_depth;
                            feature_slm_split = split;
                        }
                    }
                }
            }
            if (split * fsv >= params.weights.IFM().v)
                break;
        }
    }

    return BlockParams{ block_width, block_height, block_depth, block_features, in_block_width, in_block_height, in_block_depth, feature_slm_split };
}

float Convolution_kernel_b_fs_zyx_fsv16_imad::EstimateBlockParamsRatio(const convolution_params& params, const BlockParams& block) const {
    if (params.has_dynamic_outputs()) {
        return -10.f;
    }

    float occupancy_by_logic_size = static_cast<float>(params.outputs[0].LogicalSize() / static_cast<size_t>(params.engineInfo.maxThreadsPerDevice));
    bool increase_max_reg_pressure = occupancy_by_logic_size >= 595.f;
    bool twice_increase_max_reg_pressure = occupancy_by_logic_size >= 595.f * 2.f;
    float max_reg_pressure = twice_increase_max_reg_pressure ? 0.785f : increase_max_reg_pressure ? 0.75f : 0.7f;

    constexpr float max_occupancy = 2.f;
    constexpr float max_slm_usage = 1.f;

    // Estimate occupancy, slm usage and register pressure
    float occupancy = EstimateOccupancy(params, block);
    float slm_usage = EstimateSLMUsage(params, block);
    float reg_pressure = EstimateRegPressure(params, block);

    // Estimate fb32 usage factor
    auto& output = params.outputs[0];
    float feature_block_32 = static_cast<float>(block.output_block_features == 32);
    float fb32_factor = -5.f;
    if (params.engineInfo.deviceType == dev_type::discrete_gpu && params.engineInfo.supports_imad) {
        // Known cases where fb32 for discrete GPU works better
        bool fb32_exception_1 = output.X().v % 13 == 0 && output.X().v * output.Feature().v == 13312;
        bool fb32_exception_2 = (output.X().v % 28 == 0 && output.X().v * output.Feature().v == 14336) || (output.X().v == 14 && output.Feature().v == 512);
        bool fb32_exception_3 = (output.X().v == 5 || output.X().v == 9) && output.Feature().v == 128;
        bool fb32_exception_4 = output.X().v == 18 && output.Feature().v == 64;
        bool fb32_exception_5 = output.X().v == 37 && output.Feature().v == 512;
        bool fb32_exception_6 = output.X().v == 17 && output.Feature().v == 256;

        // Accumulate exceptions for z == 1
        bool fb32_exceptions = fb32_exception_1 || fb32_exception_2 || fb32_exception_3 || fb32_exception_4 || fb32_exception_5 || fb32_exception_6;

        // Exception for z != 1
        bool fb32_exception_z = output.X().v == output.Y().v && output.X().v % 28 == 0 && output.Z().v == 40 && output.Feature().v % 32 == 0;

        if ((output.X().v == output.Y().v && output.Z().v == 1 && fb32_exceptions) || fb32_exception_z)
            fb32_factor = 1.f;
    } else if (occupancy_by_logic_size >= 2500.f) {
        fb32_factor = 0.5f;
    }

    // We use arctangens function below for estimation of reg_pressure_factor and slm_usage_factor because arctangens
    // is a symmetric function with positive values for x > 0 (we are only interested in positive values because
    // the occupancy is always a positive value). For low occupancy (e.g. < 1) we shouldn't concentrate our attention on
    // reg_pressure_factor and slm_usage_factor because target №1 in these cases is the occupancy increase. So for occupancy < 1
    // reg_pressure factor and slm_usage_factor are enough low and they grow with growth of occupancy. Pi number (3.14159f)
    // is a scaling coefficient for setting function values in range [0; 0.5f].
    float reg_pressure_factor = atanf(occupancy) / 3.14159f;
    float slm_usage_factor = atanf(occupancy) / 3.14159f;

    size_t cur_increase_occupancy_coeff = (block.output_block_features == fsv ? 2 : 1) * block.feature_slm_split;
    size_t max_increase_occupancy_coeff = 2 * params.engineInfo.maxWorkGroupSize / simd;
    float can_increase_occupancy_coeff = static_cast<float>(max_increase_occupancy_coeff) / static_cast<float>(cur_increase_occupancy_coeff);

    // We should check if there is a possibility for increase of occupancy if occupancy is less than 1.0
    auto c_ifm_mul = CeilDiv(params.weights.IFM().v, fsv) % (params.engineInfo.maxWorkGroupSize / simd) == 0;
    auto can_increase_occupancy = (occupancy * can_increase_occupancy_coeff >= 1.0f) && c_ifm_mul;

    float reduce_occupancy = 0.0f;
    if (occupancy > max_occupancy) {
        reduce_occupancy = log10f(occupancy - max_occupancy);
        occupancy = max_occupancy;
    }

    // Estimate current block_params_ratio
    float block_params_ratio = occupancy +
                               feature_block_32 * fb32_factor +
                               slm_usage * slm_usage_factor +
                               reg_pressure * reg_pressure_factor -
                               reduce_occupancy;

    // Check all restrictions
    bool bad_block_params = reg_pressure > max_reg_pressure || slm_usage > max_slm_usage || (occupancy < 1.0f && can_increase_occupancy);

    return bad_block_params ? -10.f : block_params_ratio;
}

float Convolution_kernel_b_fs_zyx_fsv16_imad::EstimateRegPressure(const convolution_params& params, const BlockParams& block) const {
    size_t bytes_used = 0;

    // Accumulator
    size_t accumulator_elements = block.output_block_width * block.output_block_height * block.output_block_depth * block.output_block_features;
    bytes_used += accumulator_elements * BytesPerElement(GetAccumulatorType(params));

    // Input block
    size_t input_block_elements = block.input_block_depth * block.input_block_height * Align(block.input_block_width, simd) * fsv;
    bytes_used += input_block_elements * BytesPerElement(params.inputs[0].GetDType());

    // Weights block
    size_t weights_block_elements = block.output_block_features * fsv;
    bytes_used += weights_block_elements * BytesPerElement(params.weights.GetDType());

    // Experimentally selected number of registers needed for extra variables (eg. out_x, out_y, out_z, filter_idx, etc.)
    constexpr size_t experimental_extra_regs = 8 * 32;
    bytes_used += experimental_extra_regs;

    // Experimentally selected number of registers needed for slm handling
    constexpr size_t experimental_slm_regs = 4 * 32;
    if (block.feature_slm_split != 1) {
        bytes_used += experimental_slm_regs;
    }

    constexpr size_t reg_num = 128;
    constexpr size_t bytes_per_reg = 32;
    constexpr size_t max_reg_bytes = reg_num * bytes_per_reg;

    return static_cast<float>(bytes_used) / static_cast<float>(max_reg_bytes);
}

float Convolution_kernel_b_fs_zyx_fsv16_imad::EstimateOccupancy(const convolution_params& params, const BlockParams& block) const {
    size_t blocks_w = CeilDiv(params.outputs[0].X().v, block.output_block_width);
    size_t blocks_h = CeilDiv(params.outputs[0].Y().v, block.output_block_height);
    size_t blocks_d = CeilDiv(params.outputs[0].Z().v, block.output_block_depth);
    size_t blocks_f = CeilDiv(params.weights.OFM().v, block.output_block_features) * params.groups * block.feature_slm_split;
    size_t block_b = params.outputs[0].Batch().v;

    auto threads = blocks_w * blocks_h * blocks_d * blocks_f * block_b;

    return static_cast<float>(threads) / static_cast<float>(params.engineInfo.maxThreadsPerDevice);
}

float Convolution_kernel_b_fs_zyx_fsv16_imad::EstimateSLMUsage(const convolution_params& params, const BlockParams& block) const {
    if (block.feature_slm_split == 1)
        return 0.f;

    size_t slm_elements_per_work_group = block.output_block_width * block.output_block_height * block.output_block_depth *
                                         block.output_block_features * (block.feature_slm_split - 1);
    size_t slm_bytes_per_work_group = slm_elements_per_work_group * BytesPerElement(GetAccumulatorType(params));

    // Check maxLocalMemSize limitations
    size_t max_slm_bytes_per_sub_slice = params.engineInfo.maxLocalMemSize;
    if (slm_bytes_per_work_group > max_slm_bytes_per_sub_slice)
        return 0.f;

    // Estimate work groups number
    const auto& output = params.outputs[0];
    size_t work_groups_number = CeilDiv(output.X().v, block.output_block_width) *
                                CeilDiv(output.Y().v, block.output_block_height) *
                                CeilDiv(output.Z().v, block.output_block_depth) *
                                output.Batch().v *
                                CeilDiv(params.weights.OFM().v, block.output_block_features) *
                                params.groups;

    // Check work groups per device limitations
    size_t max_threads_per_compute_unit = static_cast<size_t>(params.engineInfo.maxThreadsPerExecutionUnit);
    constexpr size_t max_compute_units_per_sub_slice = 8;
    constexpr size_t max_work_groups_per_sub_slice = 16;
    size_t max_sub_slices_per_device = params.engineInfo.computeUnitsCount / max_compute_units_per_sub_slice;
    size_t max_work_groups_per_device = max_sub_slices_per_device * max_work_groups_per_sub_slice;
    if (work_groups_number > max_work_groups_per_device * 100)
        return 0.f;

    // Estimate work groups number in sub slice
    size_t threads_per_work_group = block.feature_slm_split;
    size_t threads_per_sub_slice = max_threads_per_compute_unit * max_compute_units_per_sub_slice;
    size_t current_max_work_groups_per_sub_slice = threads_per_sub_slice / threads_per_work_group;
    while (current_max_work_groups_per_sub_slice * slm_bytes_per_work_group > max_slm_bytes_per_sub_slice)
        current_max_work_groups_per_sub_slice--;

    // The best scenario for slm usage from the point of view of time spending is a case with 1 work group per sub slice
    // due to time isn't spent on waiting of synchronizations between work groups in sub slice
    if (current_max_work_groups_per_sub_slice == 1)
        return 1.0;

    // Estimate the size of the SLM memory used
    float max_slm_bytes_per_work_group = static_cast<float>(max_slm_bytes_per_sub_slice) / static_cast<float>(current_max_work_groups_per_sub_slice);
    max_slm_bytes_per_work_group = static_cast<float>(Align(static_cast<size_t>(max_slm_bytes_per_work_group), 1024));
    if (max_slm_bytes_per_work_group * static_cast<float>(current_max_work_groups_per_sub_slice) > static_cast<float>(max_slm_bytes_per_sub_slice))
        max_slm_bytes_per_work_group -= 1024.0;

    return static_cast<float>(slm_bytes_per_work_group) / static_cast<float>(max_slm_bytes_per_work_group);
}

ParamsKey Convolution_kernel_b_fs_zyx_fsv16_imad::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);

    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableGroupedConvolution();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableDilation();
    k.EnableDynamicShapesSupport();
    return k;
}

DeviceFeaturesKey Convolution_kernel_b_fs_zyx_fsv16_imad::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

KernelsData Convolution_kernel_b_fs_zyx_fsv16_imad::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

JitConstants Convolution_kernel_b_fs_zyx_fsv16_imad::GetJitConstants(const convolution_params& params,
                                                                     const DispatchData& dispatchData) const {
    auto mem_consts = Parent::GetJitConstants(params, dispatchData);

    auto block_params = GetBlockParams(params);

    bool unroll_filter_y = block_params.output_block_height != 1;
    bool unroll_filter_z = block_params.output_block_depth != 1;

    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_WIDTH", block_params.output_block_width));
    mem_consts.AddConstant(MakeJitConstant("IN_BLOCK_WIDTH", block_params.input_block_width));
    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_HEIGHT", block_params.output_block_height));
    mem_consts.AddConstant(MakeJitConstant("IN_BLOCK_HEIGHT", block_params.input_block_height));
    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_DEPTH", block_params.output_block_depth));
    mem_consts.AddConstant(MakeJitConstant("IN_BLOCK_DEPTH", block_params.input_block_depth));
    mem_consts.AddConstant(MakeJitConstant("FILTER_SIZE_Y_UNROLL", unroll_filter_y ? params.filterSize.y : 1));
    mem_consts.AddConstant(MakeJitConstant("FILTER_SIZE_Z_UNROLL", unroll_filter_z ? params.filterSize.z : 1));
    mem_consts.AddConstant(MakeJitConstant("OFM_BLOCKS_PER_SIMD", static_cast<int>(std::ceil(block_params.output_block_features / simd))));
    mem_consts.AddConstant(MakeJitConstant("OFM_SIZE_PER_SIMD", block_params.output_block_features));
    mem_consts.AddConstant(MakeJitConstant("FEATURE_SLM_SPLIT", block_params.feature_slm_split));
    mem_consts.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    mem_consts.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        std::vector<std::string> idx_order = { "out_b", "(out_f + ofb * 16)", "(out_y + oh)", "(out_x + ow)" };
        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5) {
            idx_order = { "out_b", "(out_f + ofb * 16)", "(out_z + od)", "(out_y + oh)", "(out_x + ow)" };
        }

        std::vector<Tensor::DataChannelName> loop_axes = { Tensor::DataChannelName::X };

        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5) {
            if (block_params.output_block_depth != 1) {
                loop_axes.push_back(Tensor::DataChannelName::Z);
            } else {
                idx_order[idx_order.size() - 3] = "out_z";
            }
        }

        if (block_params.output_block_height != 1) {
            loop_axes.push_back(Tensor::DataChannelName::Y);
        } else {
            idx_order[idx_order.size() - 2] = "out_y";
        }

        FusedOpsConfiguration conf_scalar = { "_SCALAR",
                                              idx_order,
                                              "dequantized_val",
                                              input_dt,
                                              1,
                                              LoadType::LT_UNALIGNED,
                                              BoundaryCheck::DISABLED };
        conf_scalar.SetLoopAxes(loop_axes, true);

        mem_consts.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return mem_consts;
}  // GetJitConstants

ConvolutionKernelBase::DispatchData Convolution_kernel_b_fs_zyx_fsv16_imad::SetDefault(const convolution_params& params,
                                                                                       int) const {
    const BlockParams& block_params = GetBlockParams(params);
    return CalcDispatchDataWithBlockParams(params, block_params);
}  // SetDefault

ConvolutionKernelBase::DispatchData Convolution_kernel_b_fs_zyx_fsv16_imad::CalcDispatchDataWithBlockParams(const convolution_params& params,
                                                                                                            const BlockParams& block_params) const {
    DispatchData dispatchData;
    const auto& output = params.outputs[0];
    const auto& weights = params.weights;

    dispatchData.gws[0] = CeilDiv(output.X().v, block_params.output_block_width);
    dispatchData.gws[1] = CeilDiv(output.Y().v, block_params.output_block_height) * CeilDiv(output.Z().v, block_params.output_block_depth);
    dispatchData.gws[2] = output.Batch().v * CeilDiv(weights.OFM().v, block_params.output_block_features) *
                          params.groups * simd * block_params.feature_slm_split;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = simd * block_params.feature_slm_split;

    dispatchData.cldnnStyle = {0, 0, 0, 0, 0};
    dispatchData.gemmStyle = {0, 0, 0, 0, 0, 0};
    dispatchData.blockParams = { block_params.output_block_width, block_params.output_block_height,
                                 block_params.output_block_depth, block_params.output_block_features,
                                 block_params.input_block_width, block_params.input_block_height,
                                 block_params.input_block_depth, block_params.feature_slm_split };
    return dispatchData;
}

KernelsPriority Convolution_kernel_b_fs_zyx_fsv16_imad::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const convolution_params&>(params);

    if (!p.is_shape_agnostic) {
        if (static_cast<float>(p.weights.IFM().v) / static_cast<float>(Align(p.weights.IFM().v, fsv)) < 0.5f)
            return FORCE_PRIORITY_4;
        else
            return FORCE_PRIORITY_2;
    } else {
        return FORCE_PRIORITY_4;
    }
}

bool Convolution_kernel_b_fs_zyx_fsv16_imad::Validate(const Params& params) const {
    if (!Parent::Validate(params)) {
        return false;
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& conv_params = *static_cast<convolution_params*>(kd.params.get());

    if (conv_params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS) {
        if ((conv_params.activations_zero_points.empty() || conv_params.weights_zero_points.empty()) &&
            (conv_params.compensation.empty()))
            return false;
    } else if (conv_params.quantization == QuantizationType::ASYMMETRIC_DATA) {
        if ((conv_params.activations_zero_points.empty()) &&
            (conv_params.compensation.empty()))
            return false;
    } else if (conv_params.quantization == QuantizationType::ASYMMETRIC_WEIGHTS) {
        if (conv_params.weights_zero_points.empty())
            return false;
    } else {
        if (!conv_params.activations_zero_points.empty() ||
            !conv_params.weights_zero_points.empty() ||
            !conv_params.compensation.empty())
            return false;
    }

    return true;
}

void Convolution_kernel_b_fs_zyx_fsv16_imad::GetUpdateDispatchDataFunc(KernelData& kd) const {
    const auto& prim_params = static_cast<const convolution_params&>(*kd.params);
    const auto& dynamicDispatchData = SetDefault(prim_params);

    kd.update_dispatch_data_func = [this, dynamicDispatchData](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const convolution_params&>(params);
        const auto& dispatchData = CalcDispatchDataWithBlockParams(prim_params, dynamicDispatchData.blockParams);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);

        kd.internalBuffers.clear();
        kd.internalBuffers.push_back(prim_params.inputs[0].PhysicalSizeInBytes());
        kd.internalBufferDataType = prim_params.inputs[0].GetDType();
    };
}

}  // namespace kernel_selector
