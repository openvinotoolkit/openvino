// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_selector_utils.h"
#include "reorder/reorder_weights_kernel_selector.h"
#include "reorder/reorder_kernel_selector.h"
#include "reorder/reorder_kernel_base.h"
#include "convolution/convolution_params.h"
#include <vector>
#include <memory>

namespace kernel_selector {

enum WeightsFormatSupportType { SUPPORTED, REORDER_NEEDED, UNSUPPORTED };

static WeightsType DataTypeToWeightsType(Datatype t) {
    switch (t) {
        case Datatype::UINT8:
            return WeightsType::UINT8;
        case Datatype::INT8:
            return WeightsType::INT8;
        case Datatype::F16:
            return WeightsType::F16;
        case Datatype::F32:
            return WeightsType::F32;
        case Datatype::INT32:
            return WeightsType::INT32;
        default:
            return WeightsType::UNSUPPORTED;
    }
}

static WeightsFormatSupportType CheckWeights(const weight_bias_params& newParams,
                                             WeightsType reqType,
                                             WeightsLayout reqLayouts,
                                             const ParamsKey& paramsKey,
                                             bool rotate) {
    // validate if weights type is image and if device supports requested sizes
    if (Tensor::IsImageType(reqLayouts)) {
        if (!CheckImageSize(newParams, reqLayouts))
            return UNSUPPORTED;
    }

    const auto& tensor = newParams.weights;
    const auto pitchesDifferFromLS = tensor.PitchesDifferFromLogicalDims();
    bool reorderNeeded = false;

    if ((reqType != tensor.GetDType()) && !(paramsKey.isEnabledDifferentInputWeightsTypes())) {
        reorderNeeded |= true;
    }

    reorderNeeded |= tensor.GetLayout() != reqLayouts;
    reorderNeeded |= rotate;

    if (reorderNeeded && !pitchesDifferFromLS && !rotate) {
        reorderNeeded = !((reqLayouts == WeightsLayout::io && tensor.GetLayout() == WeightsLayout::iyxo) ||
                          (reqLayouts == WeightsLayout::oi && tensor.GetLayout() == WeightsLayout::oiyx));
    }

    return reorderNeeded ? REORDER_NEEDED : SUPPORTED;
}

std::vector<size_t> GetImageSizes(const kernel_selector::WeightsTensor& dimensions, const WeightsLayout layout) {
    auto ofm = dimensions.OFM().v;
    auto ifm = dimensions.IFM().v;
    auto x = dimensions.X().v;
    auto y = dimensions.Y().v;

    switch (layout) {
        case WeightsLayout::image_2d_weights_c1_b_fyx:
        case WeightsLayout::image_2d_weights_c4_fyx_b:
            return {ofm, ifm * x * y};
        case WeightsLayout::image_2d_weights_winograd_6x3_s1_fbxyb:
            return {ofm * x * y * 8 / 3, ifm};
        case WeightsLayout::image_2d_weights_winograd_6x3_s1_xfbyb:
            return {ofm * y, ifm * x * 8 / 3};
        default:
            return {0, 0};
    }
}

bool CheckImageSize(const weight_bias_params& newParams, const WeightsLayout layout) {
    if (!newParams.engineInfo.supports_image)
        return false;

    auto image_sizes = GetImageSizes(newParams.weights, layout);
    if (image_sizes[0] == 0 || image_sizes[1] == 0 || image_sizes[0] > newParams.engineInfo.maxImage2dWidth ||
        image_sizes[1] > newParams.engineInfo.maxImage2dHeight)
        return false;

    return true;
}

bool UpdateWeightsParams(weight_bias_params& newParams,
                         WeightsLayout reqLayout,
                         WeightsReorderParams& weightsReorderParams,
                         const ParamsKey& paramsKey,
                         size_t groups,
                         bool rotate) {
    const auto inType = DataTypeToWeightsType(newParams.inputs[0].GetDType());
    const auto dtype = paramsKey.isEnabledDifferentInputWeightsTypes() ? newParams.weights.GetDType() : inType;
    switch (CheckWeights(newParams, inType, reqLayout, paramsKey, rotate)) {
        case SUPPORTED:
            return true;
        case UNSUPPORTED:
            return false;
        case REORDER_NEEDED: {
            if (!newParams.allowStaticInputReordering) {
                return false;
            }
            weightsReorderParams.is_initialized = true;
            weightsReorderParams.src = newParams.weights;
            weightsReorderParams.dest = newParams.weights.TransformIgnorePadding(reqLayout, dtype, groups, false);
            weightsReorderParams.rotate = rotate;

            newParams.weights = newParams.weights.TransformIgnorePadding(reqLayout, dtype, groups);
            return true;
        }
    }

    return false;
}

JitConstants GetTensorFriendlyWorkGroupsJit(const DataTensor& t) {
    auto b = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::BATCH);
    auto f = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::FEATURE);
    auto x = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::X);

    int gws_batch = -1;
    int gws_feature = -1;
    int gws_spatial = -1;

    int idx = 0;
    for (size_t i = 0; i < t.GetDims().size(); i++) {
        if (b == static_cast<int>(i))
            gws_batch = idx++;
        if (f == static_cast<int>(i))
            gws_feature = idx++;
        if (x == static_cast<int>(i))
            gws_spatial = idx++;
    }

    if (-1 == gws_batch)
        gws_batch = idx++;
    if (-1 == gws_feature)
        gws_feature = idx++;
    if (-1 == gws_spatial)
        gws_spatial = idx++;

    JitConstants jit{
        MakeJitConstant("GWS_BATCH", gws_batch),
        MakeJitConstant("GWS_FEATURE", gws_feature),
        MakeJitConstant("GWS_YX", gws_spatial),
    };

    return jit;
}

std::vector<size_t> GetTensorFriendlyWorkGroups(const DataTensor& t) {
    std::vector<size_t> sizes;
    auto x = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::X);
    auto y = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::Y);
    auto z = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::Z);
    auto w = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::W);
    auto u = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::U);
    auto v = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::V);

    auto primary_spatial_axis = x;
    if (y < primary_spatial_axis && y != -1) primary_spatial_axis = y;
    if (z < primary_spatial_axis && z != -1) primary_spatial_axis = z;
    if (w < primary_spatial_axis && w != -1) primary_spatial_axis = w;
    if (u < primary_spatial_axis && u != -1) primary_spatial_axis = u;
    if (v < primary_spatial_axis && v != -1) primary_spatial_axis = v;

    for (size_t i = 0; i < t.GetDims().size(); i++) {
        const auto& o = t.GetDims()[i];
        auto cur_axis_is_spatial = x == static_cast<int>(i) ||
                                   y == static_cast<int>(i) ||
                                   z == static_cast<int>(i) ||
                                   w == static_cast<int>(i) ||
                                   u == static_cast<int>(i) ||
                                   v == static_cast<int>(i);
        if (cur_axis_is_spatial && primary_spatial_axis != static_cast<int>(i)) {
            sizes.back() *= o.v;
        } else {
            sizes.push_back(o.v);
        }
    }

    for (size_t i = sizes.size(); i < 3; i++) {
        sizes.push_back(1U);
    }

    return sizes;
}

std::vector<size_t> GetOptimalLocalWorkGroupSizes(std::vector<size_t> gws, const EngineInfo& info,
                                                  DataLayout input_layout, DataLayout output_layout,
                                                  std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws) {
    enum axis { x, y, z, w, u, v, f, b, unused_axis };

    // GWS/LWS priority order should be considered for better local WGS setting
    // and as a result more optimized data reading/writing inside kernels
    std::vector<size_t> priority_order = { 0, 1, 2 };
    std::vector<size_t> layout_order = { x, y, z, w, u, v, f, b };

    const size_t gws_dims_num = priority_order.size();
    const size_t axis_num = layout_order.size();
    size_t first_axis_idx = 0;

    std::vector<size_t> axis_by_gws = { unused_axis, unused_axis, unused_axis, unused_axis, unused_axis, unused_axis, unused_axis, unused_axis };
    for (size_t gws_idx = 0; gws_idx < gws_dims_num; gws_idx++) {
        for (size_t axis_idx = 0; axis_idx < dims_by_gws[gws_idx].size(); axis_idx++) {
            axis_by_gws[static_cast<size_t>(dims_by_gws[gws_idx][axis_idx])] = gws_idx;
        }
    }

    auto calculate_optimized_priority_order = [&]() -> void {
        while (axis_by_gws[layout_order[first_axis_idx]] == unused_axis)
            first_axis_idx++;

        for (size_t gws_idx = 0; gws_idx < gws_dims_num; gws_idx++) {
            for (size_t axis_idx = first_axis_idx; axis_idx < axis_num; axis_idx++) {
                if (axis_by_gws[layout_order[axis_idx]] != unused_axis) {
                    bool is_already_exists = false;
                    if (axis_idx > 0) {
                        for (int i = static_cast<int>(axis_idx) - 1; i >= 0; i--) {
                            if (axis_by_gws[layout_order[axis_idx]] == axis_by_gws[layout_order[i]]) {
                                is_already_exists = true;
                                break;
                            }
                        }
                    }
                    first_axis_idx++;
                    if (!is_already_exists) {
                        priority_order[gws_idx] = axis_by_gws[layout_order[axis_idx]];
                        break;
                    }
                }
            }
        }
    };

    auto one_layout = input_layout == output_layout;

    auto simple_planar_layout = Tensor::SimpleLayout(output_layout);

    auto blocked_fsv_layout = output_layout == DataLayout::b_fs_yx_fsv2 || output_layout == DataLayout::b_fs_zyx_fsv2 ||
                              output_layout == DataLayout::b_fs_yx_fsv4 || output_layout == DataLayout::b_fs_zyx_fsv4 ||
                              output_layout == DataLayout::b_fs_yx_fsv8 || output_layout == DataLayout::b_fs_zyx_fsv8 ||
                              output_layout == DataLayout::b_fs_yx_fsv16 || output_layout == DataLayout::b_fs_zyx_fsv16 ||
                              output_layout == DataLayout::b_fs_yx_fsv32 || output_layout == DataLayout::b_fs_zyx_fsv32 ||
                              output_layout == DataLayout::fs_b_yx_fsv32;

    auto blocked_bsv_fsv_layout = output_layout == DataLayout::bs_fs_yx_bsv16_fsv2 || output_layout == DataLayout::bs_fs_zyx_bsv16_fsv2 ||
                                  output_layout == DataLayout::bs_fs_yx_bsv16_fsv4 || output_layout == DataLayout::bs_fs_zyx_bsv16_fsv4 ||
                                  output_layout == DataLayout::bs_fs_yx_bsv16_fsv8 || output_layout == DataLayout::bs_fs_zyx_bsv16_fsv8 ||
                                  output_layout == DataLayout::bs_fs_yx_bsv16_fsv16 || output_layout == DataLayout::bs_fs_yx_bsv16_fsv32 ||
                                  output_layout == DataLayout::bs_fs_yx_bsv32_fsv16 || output_layout == DataLayout::bs_fs_yx_bsv32_fsv32 ||
                                  output_layout == DataLayout::bs_fs_zyx_bsv16_fsv16 || output_layout == DataLayout::bs_fs_zyx_bsv16_fsv32 ||
                                  output_layout == DataLayout::bs_fs_zyx_bsv32_fsv16 || output_layout == DataLayout::bs_fs_zyx_bsv32_fsv32;

    auto try_change_priority_order = (simple_planar_layout || blocked_fsv_layout || blocked_bsv_fsv_layout) && one_layout;

    if (try_change_priority_order) {
        if (simple_planar_layout) {
            switch (output_layout) {
                case DataLayout::bf:
                    layout_order = { f, b, x, y, z, w, u, v };
                    break;
                case DataLayout::fb:
                    layout_order = { b, f, x, y, z, w, u, v };
                    break;
                case DataLayout::bfyx:
                    layout_order = { x, y, f, b, z, w, u, v };
                    break;
                case DataLayout::yxfb:
                    layout_order = { b, f, x, y, z, w, u, v };
                    break;
                case DataLayout::byxf:
                    layout_order = { f, x, y, b, z, w, u, v };
                    break;
                case DataLayout::byfx:
                    layout_order = { x, f, y, b, z, w, u, v };
                    break;
                case DataLayout::bxfy:
                    layout_order = { y, f, x, b, z, w, u, v };
                    break;
                case DataLayout::fbyx:
                    layout_order = { x, y, b, f, z, w, u, v };
                    break;
                case DataLayout::fyxb:
                    layout_order = { b, x, y, f, z, w, u, v };
                    break;
                case DataLayout::bfxy:
                    layout_order = { y, x, f, b, z, w, u, v };
                    break;
                case DataLayout::bfzyx:
                    layout_order = { x, y, z, f, b, w, u, v };
                    break;
                case DataLayout::bzyxf:
                    layout_order = { f, x, y, z, b, w, u, v };
                    break;
                case DataLayout::bfwzyx:
                    layout_order = { x, y, z, w, f, b, u, v };
                    break;
                case DataLayout::bfuwzyx:
                    layout_order = { x, y, z, w, u, f, b, v };
                    break;
                case DataLayout::bfvuwzyx:
                    layout_order = { x, y, z, w, u, v , f, b };
                    break;
                default:
                    layout_order = { x, y, z, w, u, v, f, b };
                    break;
            }
        } else if (blocked_fsv_layout) {
            if (output_layout == DataLayout::b_fs_yx_fsv2 || output_layout == DataLayout::b_fs_yx_fsv4 || output_layout == DataLayout::b_fs_yx_fsv8 ||
                output_layout == DataLayout::b_fs_yx_fsv16 || output_layout == DataLayout::b_fs_yx_fsv32) {
                layout_order = { f, x, y, b, z, w, u, v };
            } else if (output_layout == DataLayout::b_fs_zyx_fsv2 || output_layout == DataLayout::b_fs_zyx_fsv4 || output_layout == DataLayout::b_fs_zyx_fsv8 ||
                       output_layout == DataLayout::b_fs_zyx_fsv16 || output_layout == DataLayout::b_fs_zyx_fsv32) {
                layout_order = { f, x, y, z, b, w, u, v };
            } else { // output_layout == DataLayout::fs_b_yx_fsv32
                layout_order = { f, x, y, b, z, w, u, v };
            }
        } else if (blocked_bsv_fsv_layout) {
            layout_order = { f, b, x, y, z, w, u, v };
        }

        calculate_optimized_priority_order();

        // Revert basic priority if something is wrong
        if (priority_order[0] == priority_order[1] || priority_order[0] == priority_order[2] || priority_order[1] == priority_order[2] ||
            priority_order[0] > 2 || priority_order[1] > 2 || priority_order[2] > 2) {
            priority_order[0] = 0;
            priority_order[1] = 1;
            priority_order[2] = 2;
        }
    }

    size_t lws_max = info.maxWorkGroupSize;
    const size_t optimal_lws_values[] = { 1024, 960, 896, 832, 768, 704, 640, 576,
                                          512, 480, 448, 416, 384, 352, 320, 288,
                                          256, 227, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 2, 1 };
    const size_t suboptimal_lws_values[] = { 1024, 960, 896, 832, 768, 704, 640, 576,
                                             512, 480, 448, 416, 384, 352, 320, 288,
                                             256, 227, 224, 192, 160, 128, 96, 64, 32, 16,
                                             15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

    size_t first_lws_idx = lws_max == 1024 ? 0:
                           lws_max == 512 ?  8:
                                            16;
    // Reduces max local wgs for some cases on Gen12+ devices
    if (lws_max >= 512) {
        auto two_dims_are_odd_and_equal = (gws[0] % 2 && gws[0] > 7 && (gws[0] == gws[1] || gws[0] == gws[2])) ||
                                          (gws[1] % 2 && gws[1] > 7 && gws[1] == gws[2]);

        // Known cases when lws_max = 256 works better than lws_max > 256
        auto max_wgs_exception1 = gws[priority_order[0]] == 1278 && gws[priority_order[1]] == 718 && gws[priority_order[2]] % 10 == 0;
        auto max_wgs_exception2 = gws[priority_order[0]] == 28 && gws[priority_order[1]] == 168 && gws[priority_order[2]] == 128;
        auto max_wgs_exception3 = gws[priority_order[0]] == 1000 && gws[priority_order[1]] == 1 && gws[priority_order[2]] == 64;
        auto max_wgs_exception4 = gws[priority_order[0]] == 180 && gws[priority_order[1]] == 320 && gws[priority_order[2]] == 56;
        auto max_wgs_exception5 = gws[priority_order[0]] == 1 && gws[priority_order[1]] > 256 && gws[priority_order[2]] == 1;
        auto max_wgs_exception6 = gws[priority_order[0]] == 64 && gws[priority_order[1]] == 16 && gws[priority_order[2]] == 1 &&
                                  priority_order[1] == 2 && priority_order[2] == 1;
        if (two_dims_are_odd_and_equal || max_wgs_exception1 || max_wgs_exception2 || max_wgs_exception3 || max_wgs_exception4 ||
            max_wgs_exception5 || max_wgs_exception6) {
            lws_max = 256;
            first_lws_idx = 16;
        }
    }

    size_t total_lws = 1;
    size_t total_gws = 1;
    std::vector<size_t> lws = { 1, 1, 1 };

    for (size_t i = 0; i < gws.size(); ++i) {
        auto rest_lws = lws_max / total_lws;
        size_t lws_idx = first_lws_idx;
        size_t max_optimal_lws0_value = lws_max;
        if (try_change_priority_order && axis_by_gws[f] != unused_axis) {
            if (output_layout == DataLayout::b_fs_yx_fsv16 || output_layout == DataLayout::b_fs_zyx_fsv16 || output_layout == DataLayout::fs_b_yx_fsv32) {
                max_optimal_lws0_value = 16;
            } else if (output_layout == DataLayout::b_fs_yx_fsv32 || output_layout == DataLayout::b_fs_zyx_fsv32) {
                max_optimal_lws0_value = 32;
            } else if ((output_layout == DataLayout::bs_fs_yx_bsv16_fsv16 || output_layout == DataLayout::bs_fs_zyx_bsv16_fsv16) &&
                       (axis_by_gws[b] == axis_by_gws[f])) {
                max_optimal_lws0_value = 256;
            } else if ((output_layout == DataLayout::bs_fs_yx_bsv16_fsv16 || output_layout == DataLayout::bs_fs_zyx_bsv16_fsv16) &&
                       (axis_by_gws[b] != axis_by_gws[f]) && (axis_by_gws[b] != unused_axis)) {
                max_optimal_lws0_value = 16;
            } else if ((output_layout == DataLayout::bs_fs_yx_bsv32_fsv32 || output_layout == DataLayout::bs_fs_zyx_bsv32_fsv32) &&
                       (axis_by_gws[b] != axis_by_gws[f]) && (axis_by_gws[b] != unused_axis)) {
                max_optimal_lws0_value = 32;
            }
        }

        auto can_use_suboptimal_lws1 = (i == 1) && ((gws[priority_order[0]] % 32 == 0) || (gws[priority_order[0]] == 1 && gws[priority_order[2]] % 16 != 0));
        auto can_use_suboptimal_lws2 = (i == 2) && (total_lws == total_gws);
        const size_t* lws_values = can_use_suboptimal_lws1 || can_use_suboptimal_lws2 ?
                                   suboptimal_lws_values :
                                   optimal_lws_values;

        while (rest_lws < lws_values[lws_idx]) lws_idx++;
        if (i == 0) {
            while (lws_values[lws_idx] > max_optimal_lws0_value) lws_idx++;
        }
        while (gws[priority_order[i]] % lws_values[lws_idx]) lws_idx++;

        // else statement cannot be interpreted, it causes dg2 perf degradation, so added dg2(1024 lws_max) in if statement
        if (lws_max == 256 || lws_max == 1024 || total_lws == total_gws) {
            lws[priority_order[i]] = lws_values[lws_idx];
        } else {
            lws[priority_order[i]] = i == 2 && gws[priority_order[0]] != 1 ? 1 : lws_values[lws_idx];
            if (total_gws > 100 && total_lws < 8 && i == 2)
                lws[priority_order[i]] = lws_values[lws_idx];
        }

        total_lws *= lws_values[lws_idx];
        total_gws *= gws[priority_order[i]];
    }

    // For cases with lws { 1, 1, 1 } try to use suboptimal values to increase work group size
    if (lws[0] == 1 && lws[1] == 1 && lws[2] == 1) {
        total_lws = 1;
        for (size_t i = 0; i < gws.size(); ++i) {
            auto rest_lws = lws_max / total_lws;
            size_t lws_idx = first_lws_idx;

            const size_t* lws_values = suboptimal_lws_values;

            while (rest_lws < lws_values[lws_idx]) lws_idx++;
            while (gws[priority_order[i]] % lws_values[lws_idx]) lws_idx++;

            lws[priority_order[i]] = lws_values[lws_idx];

            total_lws *= lws_values[lws_idx];
        }
    }

    return lws;
}

bool CheckInputsOutputNoPitchSameDims(const base_params& params) {
    bool no_pitch_same_dims = true;

    std::map<DataLayout, std::pair<int, int>> block_layouts {
        {DataLayout::b_fs_yx_fsv16,          {1, 16}},
        {DataLayout::b_fs_zyx_fsv16,         {1, 16}},
        {DataLayout::b_fs_yx_fsv32,          {1, 32}},
        {DataLayout::b_fs_zyx_fsv32,         {1, 32}},
        {DataLayout::bs_fs_yx_bsv16_fsv8,    {16, 8}},
        {DataLayout::bs_fs_yx_bsv16_fsv16,   {16, 16}},
        {DataLayout::bs_fs_yx_bsv16_fsv32,   {16, 32}},
        {DataLayout::bs_fs_zyx_bsv16_fsv8,   {16, 8}},
        {DataLayout::bs_fs_zyx_bsv16_fsv16,  {16, 16}},
        {DataLayout::bs_fs_zyx_bsv16_fsv32,  {16, 32}},
        {DataLayout::bs_f_bsv8__af8,         {8, 8}},
        {DataLayout::bs_f_bsv16__af8,        {16, 8}},
        {DataLayout::b_fs_yx_fsv4,           {1, 4}},
        {DataLayout::b_fs_zyx_fsv4,          {1, 4}},
        {DataLayout::b_fs_yx_fsv8,           {1, 8}},
        {DataLayout::b_fs_zyx_fsv8,          {1, 8}},
        {DataLayout::fs_b_yx_fsv32,          {1, 32}},
        {DataLayout::bs_fs_yx_bsv32_fsv16,   {32, 16}},
        {DataLayout::bs_fs_zyx_bsv32_fsv16,  {32, 16}},
        {DataLayout::bs_fs_yx_bsv32_fsv32,   {32, 32}},
        {DataLayout::bs_fs_zyx_bsv32_fsv32,  {32, 32}}
    };

    if (params.inputs.size()) {
        no_pitch_same_dims = !params.inputs[0].PitchesDifferFromLogicalDims();

        auto block_layout = block_layouts.find(params.inputs[0].GetLayout());
        if (block_layout != block_layouts.end()) {
            auto block_size = block_layout->second;
            if (params.inputs[0].Batch().v % block_size.first != 0 || params.inputs[0].Feature().v % block_size.second != 0)
                    return false;
        }

        if (params.fused_ops.size()) {
            for (auto fused_op : params.fused_ops) {
                for (size_t in = 0; in < fused_op.tensors.size(); in++) {
                    if (fused_op.tensors[in].LogicalSize() == 1)
                        continue;

                    auto layout = block_layouts.find(fused_op.tensors[in].GetLayout());
                    if (layout != block_layouts.end()) {
                        auto block_size = layout->second;
                        if (fused_op.tensors[in].Batch().v % block_size.first != 0 || fused_op.tensors[in].Feature().v % block_size.second != 0)
                            return false;
                    }

                    no_pitch_same_dims = no_pitch_same_dims && (params.inputs[0] == fused_op.tensors[in]);
                }
            }
        }

        for (size_t i = 1; i < params.inputs.size(); i++) {
            no_pitch_same_dims = no_pitch_same_dims && (params.inputs[0] == params.inputs[i]);

            auto layout = block_layouts.find(params.inputs[i].GetLayout());
            if (layout != block_layouts.end()) {
                auto block_size = layout->second;
                if (params.inputs[i].Batch().v % block_size.first != 0 || params.inputs[i].Feature().v % block_size.second != 0)
                    return false;
            }
        }
        // TODO : check for multiple outputs
        no_pitch_same_dims = no_pitch_same_dims && (params.inputs[0] == params.outputs[0]);
    }

    return no_pitch_same_dims;
}
}  // namespace kernel_selector
