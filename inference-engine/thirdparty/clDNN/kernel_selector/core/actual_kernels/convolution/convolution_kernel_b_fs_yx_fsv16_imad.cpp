// Copyright (c) 2020 Intel Corporation
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

#include "convolution_kernel_b_fs_yx_fsv16_imad.h"
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

Convolution_kernel_b_fs_yx_fsv16_imad::BlockParams
Convolution_kernel_b_fs_yx_fsv16_imad::GetBlockParams(const convolution_params& params) const {
    constexpr float max_reg_pressure = 0.75f;

    // TODO Investigate whether below algorithm for selecting optimal block params could be reduced to:
    //      1. Enumerate possible block params as optimization space
    //      2. Prune invalid params (too high register pressure, too big local memory usage)
    //      3. Rank params according to some combination of:
    //         - compute/memory ratio
    //         - occupancy
    //         - register pressure
    //         - local memory usage
    //      4. Select params with highest rank

    // Select optimal block width
    size_t block_width = getOutBlock_X(params.output.X().v, params.stride.x, params.filterSize.x, params.dilation.x);
    size_t in_block_width = (block_width - 1) * params.stride.x + (params.filterSize.x - 1) * params.dilation.x + 1;

    // If possible increase features block size
    size_t block_features = simd;
    {
        size_t tmp_block_features = simd * 2;
        auto block2_params = BlockParams{ block_width, 1, tmp_block_features, in_block_width, 1, 1 };

        bool c_mul_f = params.output.Feature().v % tmp_block_features == 0;
        bool c_reg_pressure = EstimateRegPressure(params, block2_params) <= max_reg_pressure;

        if (c_mul_f && c_reg_pressure) {
            block_features = tmp_block_features;
        }
    }

    // If not enough occupancy try to perform feature split or/and block reduction
    size_t feature_slm_split = 1;
    auto no_split_params = BlockParams{ block_width, 1, block_features, in_block_width, 1, 1 };
    if (EstimateOccupancy(params, no_split_params) < 1.f) {
        // Temporary variables for possible reductions in block sizes
        bool update_block_params = false;
        size_t split_block_width = block_width;
        size_t split_in_block_width = in_block_width;
        size_t split_block_features = block_features;

        // Feature split requires extra registers, so check if it can be done with current block sizes
        bool can_split =
            EstimateRegPressure(params, BlockParams{ block_width, 1, block_features, in_block_width, 1, 2 }) <= max_reg_pressure;
        // Has the occupancy reached sufficient level
        bool enough_occupancy = false;
        // Reductions to reduce register pressure
        // Try to reduce block width to free some registers. Good compute/memory ratio will be pointless if barely any threads will run.
        if (!can_split && block_width != 1) {
            // At most twice reduction in output block width is acceptable
            for (size_t w = block_width; w >= CeilDiv(block_width, 2); w -= 1) {
                size_t tmp_in_width = (w - 1) * params.stride.x + (params.filterSize.x - 1) * params.dilation.x + 1;
                auto dummy_split_params = BlockParams{ w, 1, block_features, tmp_in_width, 1, 2 };

                bool c_reg_pressure = EstimateRegPressure(params, dummy_split_params) <= max_reg_pressure;
                bool c_mul_x = params.output.X().v % w == 0;

                if (c_reg_pressure && c_mul_x) {
                    split_block_width = w;
                    split_in_block_width = tmp_in_width;
                    can_split = true;
                    break;
                }
            }
        }
        // Try to reduce block features.
        // Done after attempting block width reduction, because bigger feature block allows more threads to write results in parallel.
        if (!can_split) {
            if (block_features / simd % 2 == 0) {
                split_block_features = block_features / 2;
                can_split = true;
            }
        }
        // Check if previous reductions haven't improved occupancy enough
        {
            auto reduced_params = BlockParams{ split_block_width, 1, split_block_features, split_in_block_width, 1, 1 };
            enough_occupancy = EstimateOccupancy(params, reduced_params) >= 1.f;
            update_block_params = enough_occupancy;
        }

        if (can_split && !enough_occupancy) {
            // TODO Try other split sizes
            for (size_t split = 4; split < 5; ++split) {
                auto tmp_params = BlockParams{ block_width, 1, block_features, in_block_width, 1, split };

                bool c_ifm_mul = CeilDiv(params.weights.IFM().v, fsv) % split == 0;
                bool c_slm = EstimateSLMUsage(params, tmp_params) <= 1.f;
                bool c_lws = split * simd <= params.engineInfo.maxWorkGroupSize;
                bool c_reg_pressure = EstimateRegPressure(params, tmp_params) <= max_reg_pressure;
                bool c_occupancy = EstimateOccupancy(params, tmp_params) >= 1.f;

                if (c_ifm_mul && c_slm && c_lws && c_reg_pressure) {
                    feature_slm_split = split;
                    update_block_params = true;
                    enough_occupancy = c_occupancy;
                }

                // slm usage and work group sizes will only grow with split, so no point in checking
                if (!c_slm || !c_lws || split * fsv >= params.weights.IFM().v)
                    break;
            }
        }
        // Splitting was not sufficient or couldn't be done
        // Try to reduce block width if hasn't been done before
        if (!enough_occupancy && split_block_width == block_width && block_width != 1) {
            // At most twice reduction in output block width is acceptable
            for (size_t w = block_width; w >= CeilDiv(block_width, 2); w -= 1) {
                size_t tmp_in_width = (w - 1) * params.stride.x + (params.filterSize.x - 1) * params.dilation.x + 1;
                auto tmp_params = BlockParams{ w, 1, split_block_features, tmp_in_width, 1, feature_slm_split };

                bool c_occupancy = EstimateOccupancy(params, tmp_params) >= 1.f;
                bool c_mul_x = params.output.X().v % w == 0;

                if (c_mul_x) {
                    split_block_width = w;
                    split_in_block_width = tmp_in_width;
                    update_block_params = true;
                }
                // Reached enough occupancy, don't reduce futher to not hurt compute/mem ratio
                if (c_mul_x && c_occupancy)
                    break;
            }
        }
        if (update_block_params) {
            block_width = split_block_width;
            in_block_width = split_in_block_width;
            block_features = split_block_features;
        }
    }

    // Select biggest block height that fits into registers
    size_t block_height = 1;
    size_t in_block_height = 1;
    for (size_t h = 2; h < 16; ++h) {
        if (params.output.Y().v % h != 0)
            continue;

        size_t tmp_in_block_height = (h - 1) * params.stride.y + (params.filterSize.y - 1) * params.dilation.y + 1;
        auto tmp_params = BlockParams{ block_width, h, block_features, in_block_width, tmp_in_block_height, feature_slm_split };

        bool c_reg_pressure = EstimateRegPressure(params, tmp_params) <= max_reg_pressure;
        bool c_occupancy = EstimateOccupancy(params, tmp_params) >= 1.f;
        bool c_slm = EstimateSLMUsage(params, tmp_params) <= 1.f;

        if (c_reg_pressure && c_occupancy && c_slm) {
            block_height = h;
            in_block_height = tmp_in_block_height;
        } else {
            break;
        }
    }

    return BlockParams{ block_width, block_height, block_features, in_block_width, in_block_height, feature_slm_split };
}

float Convolution_kernel_b_fs_yx_fsv16_imad::EstimateRegPressure(const convolution_params& params, const BlockParams& block) const {
    size_t bytes_used = 0;
    // accumulator
    size_t accumulator_elements = block.output_block_width * block.output_block_height * block.output_block_features;
    bytes_used += accumulator_elements * BytesPerElement(GetAccumulatorType(params));
    // input block
    size_t input_block_elements = block.input_block_height * Align(block.input_block_width, simd) * fsv;
    bytes_used += input_block_elements * BytesPerElement(params.inputs[0].GetDType());
    // weights block
    size_t weights_block_elements = block.output_block_features * fsv;
    bytes_used += weights_block_elements * BytesPerElement(params.weights.GetDType());

    // Experimentally selected number of registers needed for extra variables (eg. out_x, out_y, filter_idx, etc.)
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

float Convolution_kernel_b_fs_yx_fsv16_imad::EstimateOccupancy(const convolution_params& params, const BlockParams& block) const {
    size_t blocks_w = CeilDiv(params.output.X().v, block.output_block_width);
    size_t blocks_h = CeilDiv(params.output.Y().v, block.output_block_height);
    size_t blocks_f = CeilDiv(params.output.Feature().v, block.output_block_features) * block.feature_slm_split;
    size_t block_b = params.output.Batch().v;

    auto threads = blocks_w * blocks_h * blocks_f * block_b;
    constexpr size_t max_threads_per_cu = 7;
    size_t compute_units = params.engineInfo.computeUnitsCount;
    size_t max_threads = compute_units * max_threads_per_cu;

    return static_cast<float>(threads) / static_cast<float>(max_threads);
}

float Convolution_kernel_b_fs_yx_fsv16_imad::EstimateSLMUsage(const convolution_params& params, const BlockParams& block) const {
    size_t slm_elements = block.output_block_width * block.output_block_height * block.output_block_features * (block.feature_slm_split - 1);
    size_t slm_bytes = slm_elements * BytesPerElement(GetAccumulatorType(params));

    // TODO Actual maximum slm should also depend on number of work-groups, but this is device specific
    size_t max_slm_bytes = params.engineInfo.maxLocalMemSize;

    return static_cast<float>(slm_bytes) / static_cast<float>(max_slm_bytes);
}

ParamsKey Convolution_kernel_b_fs_yx_fsv16_imad::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);

    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableDilation();
    k.DisableTuning();
    return k;
}

KernelsData Convolution_kernel_b_fs_yx_fsv16_imad::GetKernelsData(const Params& params,
                                                                   const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

JitConstants Convolution_kernel_b_fs_yx_fsv16_imad::GetJitConstants(const convolution_params& params,
                                                                     const DispatchData& kd) const {
    auto mem_consts = Parent::GetJitConstants(params, kd);

    auto block_params = GetBlockParams(params);

    bool unroll_filter_y = block_params.output_block_height != 1;

    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_WIDTH", block_params.output_block_width));
    mem_consts.AddConstant(MakeJitConstant("IN_BLOCK_WIDTH", block_params.input_block_width));
    mem_consts.AddConstant(MakeJitConstant("OUT_BLOCK_HEIGHT", block_params.output_block_height));
    mem_consts.AddConstant(MakeJitConstant("IN_BLOCK_HEIGHT", block_params.input_block_height));
    mem_consts.AddConstant(MakeJitConstant("FILTER_SIZE_Y_UNROLL", unroll_filter_y ? params.filterSize.y : 1));
    mem_consts.AddConstant(MakeJitConstant("OFM_BLOCKS_PER_SIMD", block_params.output_block_features / simd));
    mem_consts.AddConstant(MakeJitConstant("OFM_SIZE_PER_SIMD", block_params.output_block_features));
    mem_consts.AddConstant(MakeJitConstant("FEATURE_SLM_SPLIT", block_params.feature_slm_split));
    mem_consts.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    mem_consts.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        std::vector<std::string> idx_order = { "out_b", "(out_f + ofb * 16)", "(out_y + oh)", "(out_x + ow)" };
        std::vector<Tensor::DataChannelName> loop_axes = { Tensor::DataChannelName::X };
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

ConvolutionKernelBase::DispatchData Convolution_kernel_b_fs_yx_fsv16_imad::SetDefault(const convolution_params& params,
                                                                           int) const {
    DispatchData kd;
    const auto& output = params.output;
    auto block_params = GetBlockParams(params);

    kd.gws0 = CeilDiv(output.X().v, block_params.output_block_width);
    kd.gws1 = CeilDiv(output.Y().v, block_params.output_block_height);
    kd.gws2 = output.Batch().v * CeilDiv(output.Feature().v, block_params.output_block_features) * simd * block_params.feature_slm_split;

    kd.lws0 = 1;
    kd.lws1 = 1;
    kd.lws2 = simd * block_params.feature_slm_split;

    kd.cldnnStyle = {0, 0, 0, 0, 0};
    kd.gemmStyle = {0, 0, 0, 0, 0, 0};

    kd.efficiency = FORCE_PRIORITY_2;
    // TODO Optimize 1x1, because this kernel is better in most cases
    //if (params.filterSize.x == 1 && params.filterSize.y == 1)
    //    kd.efficiency = FORCE_PRIORITY_1;
    if (static_cast<float>(params.weights.IFM().v) / static_cast<float>(Align(params.weights.IFM().v, fsv)) < 0.5f)
        kd.efficiency = FORCE_PRIORITY_4;

    return kd;
}  // SetDefault

bool Convolution_kernel_b_fs_yx_fsv16_imad::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options)) {
        return false;
    }

    KernelData kd = KernelData::Default<convolution_params>(params);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    if (newParams.groups != 1 || newParams.split != 1)
        return false;

    return true;
}
}  // namespace kernel_selector
