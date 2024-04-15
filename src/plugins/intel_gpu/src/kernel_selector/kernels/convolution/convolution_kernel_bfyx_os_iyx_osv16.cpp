// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_bfyx_os_iyx_osv16.h"
#include <vector>
#include <utility>
#include <algorithm>

namespace kernel_selector {
// Sub-group size used by "kernel_name_bfyx_os_iyx_osv16" kernel.

ConvolutionKernel_bfyx_os_iyx_osv16::ConvolutionKernel_bfyx_os_iyx_osv16()
    : ConvolutionKernelBase("convolution_gpu_bfyx_os_iyx_osv16") {
    // Generate the dispatch options to the auto-tuner.
    std::vector<size_t> blockWidthSizes = {1, 2, 4, 5, 6, 8, 10, 12, 14, 16};
    std::vector<size_t> blockHeightSizes = {1, 2, 3, 4, 5};
    std::vector<size_t> prefetchSizes = {1, 2, 3, 4, 5, 6, 8, 10};
    std::vector<std::string> executionModes = ConvolutionKernelBase::autoTuneOptions;
    const size_t maxBlockSize = 60;

    for (auto executionMode : executionModes) {
        for (auto blockWidth : blockWidthSizes) {
            for (auto blockHeight : blockHeightSizes) {
                for (auto prefetch : prefetchSizes) {
                    if (blockWidth * blockHeight <= maxBlockSize) {
                        autoTuneOptions.emplace_back(AutoTuneOption{blockWidth, blockHeight, prefetch, executionMode});
                    }
                }
            }
        }
    }
}

ParamsKey ConvolutionKernel_bfyx_os_iyx_osv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableDilation();
    k.EnableGroupedConvolution();
    k.EnableDynamicShapesSupport();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_bfyx_os_iyx_osv16::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_subgroup_shuffle();
    k.requires_reqd_subgroup_size();

    return k;
}

static std::pair<size_t, size_t> get_bfyx_req_input_block_dims(size_t output_block_width,
                                                               size_t output_block_height,
                                                               const uSize& filter_size,
                                                               const uSize& stride,
                                                               const uSize& dilation,
                                                               size_t sg_size = 16,
                                                               size_t read_chunk_size = 8,
                                                               size_t min_read_size = 16) {
    assert(output_block_width > 0 && output_block_height > 0);
    assert(stride.x > 0 && stride.y > 0);
    assert(filter_size.x > 0 && filter_size.y > 0);

    // Number of elements in X dimension needed from input to compute output block without re-reading input.
    size_t input_block_req_width = (output_block_width - 1) * stride.x + (filter_size.x - 1) * dilation.x + 1;
    // Number of elements in Y dimension needed from input to compute output block without re-reading input.
    size_t input_block_req_height = (output_block_height - 1) * stride.y + (filter_size.y - 1) * dilation.y + 1;

    // Required number of elements in X dimension rounded to nearest >= read chunk size.
    size_t input_block_read_width = std::max(RoundUp(input_block_req_width, read_chunk_size), min_read_size);
    // Number of sub-group-sized vectors of unit type needed to store input block.
    size_t input_block_array_size = CeilDiv(input_block_req_height * input_block_read_width, sg_size);

    return std::make_pair(input_block_array_size, input_block_read_width);
}

static void shrink_blocks_to_output_size(size_t output_x, size_t output_y, size_t& block_x, size_t& block_y, size_t sub_group_size) {
    // how many elements we will compute in each dimension
    size_t computed_x = Align(output_x, block_x);
    size_t computed_y = Align(output_y, block_y);
    // how many simds we need in each dimension
    size_t simds_x = computed_x / block_x;
    size_t simds_y = computed_y / block_y;
    // how many unused values we have in each dimension
    size_t unused_x = computed_x - output_x;
    size_t unused_y = computed_y - output_y;

    block_x -= unused_x / simds_x;
    block_y -= unused_y / simds_y;

    if (simds_x * simds_y >= sub_group_size) {
        block_x = Align(block_x, 2);
        block_y = Align(block_y, 2);
    }
}

ConvolutionKernel_bfyx_os_iyx_osv16::AutoTuneOption ConvolutionKernel_bfyx_os_iyx_osv16::GetAutoTuneOptions(
    const Params& p,
    int autoTuneIndex) const {
    if ((autoTuneIndex >= 0) && (autoTuneIndex < static_cast<int>(autoTuneOptions.size()))) {
        return autoTuneOptions[autoTuneIndex];
    }

    AutoTuneOption option = {0, 0, 0, EXE_MODE_DEFAULT};

    const convolution_params& cp = static_cast<const convolution_params&>(p);

    const auto& sub_group_size = GetSubGroupSize(cp);

    if (cp.stride.x == 1 && cp.stride.y == 1) {
        if (cp.filterSize.x == 1 && cp.filterSize.y == 1) {
            option.blockWidth = sub_group_size;
            option.blockHeight = 1;
            option.prefetch = 4;
        // if less than 16 values is required to compute one single row of output
        // then each WI shall compute one single row to maximize reuse within SIMD subgroup (this gives very nice
        // performance results)
        } else if (!p.is_shape_agnostic && cp.outputs[0].X().v + (cp.filterSize.x - 1) * cp.dilation.x < sub_group_size) {
            option.blockWidth = cp.outputs[0].X().v;
            option.blockHeight = 1;
            option.prefetch = 4;
        } else if (cp.filterSize.x < 5 && cp.filterSize.y < 5) {
            option.blockWidth = sub_group_size - cp.filterSize.x + 1;
            option.blockHeight = 2;
            option.prefetch = 4;
        } else {
            option.blockWidth = 4;
            option.blockHeight = 3;
            option.prefetch = 4;
        }
    } else if (cp.stride.x == 2 && cp.stride.y == 2) {
        option.blockWidth = 5;
        option.blockHeight = 4;
        option.prefetch = 4;
    } else {
        option.blockWidth = 4;
        option.blockHeight = 3;
        option.prefetch = 5;
    }

    // if this is not 1x1 batch1 case then shrink filters, other way we're memory bound and it's best to use 16x1 block
    // sizes
    if (!p.is_shape_agnostic && (cp.filterSize.x != 1 || cp.filterSize.y != 1 || cp.outputs[0].Batch().v != 1)) {
        shrink_blocks_to_output_size(cp.outputs[0].X().v, cp.outputs[0].Y().v, option.blockWidth, option.blockHeight, sub_group_size);
    }
    return option;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_os_iyx_osv16::SetDefault(const convolution_params& cp,
                                                                                    int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(cp);
    const auto& sub_group_size = GetSubGroupSize(cp);

    const auto of_maps = cp.outputs[0].Feature().v;
    const auto of_maps_per_group = of_maps / cp.groups;
    const size_t of_threads_per_batch = RoundUp(of_maps_per_group, sub_group_size) * cp.groups;

    auto tuneOptions = GetAutoTuneOptions(cp, autoTuneIndex);
    dispatchData.cldnnStyle.blockWidth = tuneOptions.blockWidth;
    dispatchData.cldnnStyle.blockHeight = tuneOptions.blockHeight;
    dispatchData.cldnnStyle.prefetch = tuneOptions.prefetch;

    auto input_block_dims = get_bfyx_req_input_block_dims(dispatchData.cldnnStyle.blockWidth,
                                                          dispatchData.cldnnStyle.blockHeight,
                                                          cp.filterSize,
                                                          cp.stride,
                                                          cp.dilation,
                                                          sub_group_size,
                                                          cp.outputs[0].GetDType() == Datatype::F16 ? sub_group_size : sub_group_size / 2,
                                                          sub_group_size);
    dispatchData.cldnnStyle.inputBlockArraySize = input_block_dims.first;
    dispatchData.cldnnStyle.inputBlockWidth = input_block_dims.second;

    dispatchData.gws[0] = CeilDiv(cp.outputs[0].X().v, dispatchData.cldnnStyle.blockWidth);
    dispatchData.gws[1] = CeilDiv(cp.outputs[0].Y().v, dispatchData.cldnnStyle.blockHeight);
    dispatchData.gws[2] = of_threads_per_batch * cp.outputs[0].Batch().v;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = sub_group_size;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_bfyx_os_iyx_osv16::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}

bool ConvolutionKernel_bfyx_os_iyx_osv16::Validate(const Params& p) const {
    if (!ConvolutionKernelBase::Validate(p) || !ConvolutionCheckInput(p)) {
        return false;
    }

    // To prevent big sized filter which causes lots of CL build time.
    const size_t acceptable_filter_size = 1024;     // This acceptable size was decided by heuristics
    const auto& params = static_cast<const convolution_params&>(p);
    auto filter_size = params.filterSize.x * params.filterSize.y;
    if (filter_size > acceptable_filter_size) {
        return false;
    }

    return true;
}

JitConstants ConvolutionKernel_bfyx_os_iyx_osv16::GetJitConstants(const convolution_params& params,
                                                                  const DispatchData& dispatchData) const {
    const convolution_params& cp = static_cast<const convolution_params&>(params);
    const auto& sub_group_size = GetSubGroupSize(cp);

    const auto of_maps = params.outputs[0].Feature().v;
    const auto of_maps_per_group = of_maps / params.groups;
    const size_t of_threads_per_batch = RoundUp(of_maps_per_group, sub_group_size);
    size_t leftovers = of_threads_per_batch - of_maps_per_group;

    auto jit = Parent::GetJitConstantsWithLoopUnroll(params, dispatchData);

    if (!params.fused_ops.empty()) {
        auto input_dt = GetUnitType(params);
        FusedOpsConfiguration conf_scalar = {"", {"batch_idx", "feature_num", "(or+r)", "(oc+c)"}, "dst", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    jit.AddConstant(MakeJitConstant("OSV_SIZE", 16));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", dispatchData.lws[2]));
    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_WIDTH", dispatchData.cldnnStyle.blockWidth));
    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_HEIGHT", dispatchData.cldnnStyle.blockHeight));
    jit.AddConstant(MakeJitConstant("IN_BLOCK_ARRAY_SIZE", dispatchData.cldnnStyle.inputBlockArraySize));
    jit.AddConstant(MakeJitConstant("IN_BLOCK_WIDTH", dispatchData.cldnnStyle.inputBlockWidth));
    jit.AddConstant(MakeJitConstant("PREFETCH", dispatchData.cldnnStyle.prefetch));

    if (leftovers) {
        jit.AddConstant(MakeJitConstant("LEFTOVERS", leftovers));
    }

    return jit;
}

WeightsLayout ConvolutionKernel_bfyx_os_iyx_osv16::GetPreferredWeightsLayout(
        const convolution_params &params) const {
    return (params.groups > 1) ? WeightsLayout::g_os_iyx_osv16 : WeightsLayout::os_iyx_osv16;
}

KernelsData ConvolutionKernel_bfyx_os_iyx_osv16::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}

KernelsData ConvolutionKernel_bfyx_os_iyx_osv16::GetKernelsDataForAutoTune(const Params& params) const {
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
