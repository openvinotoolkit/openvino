// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <kernel_selector_utils.h>
#include "resample_kernel_ref.h"

#include <algorithm>
#include <vector>
#include <string>

namespace kernel_selector {

ParamsKey ResampleKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableResampleType(ResampleType::NEAREST_NEIGHBOR);
    k.EnableResampleType(ResampleType::CAFFE_BILINEAR_INTERP);
    k.EnableResampleType(ResampleType::BILINEAR_INTERP);
    k.EnableResampleType(ResampleType::CUBIC);
    k.EnableResampleType(ResampleType::LINEAR_ONNX);
    return k;
}

KernelsData ResampleKernelRef::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

static size_t packing_factor(const resample_params& params) {
    // TODO Add support for only input packing
    bool in_out_8bit = (params.inputs[0].GetDType() == Datatype::UINT8 || params.inputs[0].GetDType() == Datatype::INT8) &&
                       (params.outputs[0].GetDType() == Datatype::UINT8 || params.outputs[0].GetDType() == Datatype::INT8);

    if (!in_out_8bit)
        return 1;

    auto get_layout_packing_factor = [](const DataLayout& layout) -> size_t {
        switch (layout) {
        case DataLayout::b_fs_yx_fsv16:
        case DataLayout::bs_fs_yx_bsv32_fsv16:
            return 16;
        case DataLayout::b_fs_yx_fsv4:
            return 4;
        default:
            break;
        }
        return 1;
    };

    size_t input_factor = get_layout_packing_factor(params.inputs[0].GetLayout());
    size_t output_factor = get_layout_packing_factor(params.outputs[0].GetLayout());

    if (input_factor % output_factor == 0 || output_factor % input_factor == 0)
        return std::min(input_factor, output_factor);
    return 1;
}

static bool use_packing(const resample_params& params) {
    if (params.resampleType != ResampleType::NEAREST_NEIGHBOR)
        return false;

    auto pack = packing_factor(params);
    if (pack == 1)
        return false;

    if (params.inputs[0].Feature().pad.before % pack != 0 || params.outputs[0].Feature().pad.before % pack != 0)
        return false;

    auto packed_work_items = params.outputs[0].X().v * params.outputs[0].Y().v * params.outputs[0].Z().v
        * CeilDiv(params.outputs[0].Feature().v, pack) * params.outputs[0].Batch().v;
    // TODO Loosen this requirement to minimum EUs needed to saturate cache bandwidth
    size_t max_work_items_per_eu = 32 * static_cast<size_t>(params.engineInfo.maxThreadsPerExecutionUnit);
    auto minimum_work_items = params.engineInfo.computeUnitsCount * max_work_items_per_eu;

    if (packed_work_items < minimum_work_items)
        return false;

    return true;
}

static bool has_padding(const resample_params& params) {
    return std::any_of(params.pads_begin.begin(), params.pads_begin.end(), [](const auto pad) { return pad != 0; }) ||
           std::any_of(params.pads_end.begin(), params.pads_end.end(), [](const auto pad) { return pad != 0; });
}

static bool is_integral_ratio(size_t lhs, size_t rhs) {
    return lhs != 0 && rhs != 0 && (lhs % rhs == 0 || rhs % lhs == 0);
}

static bool is_integral_upsampling_ratio(size_t output, size_t input) {
    return input != 0 && output >= input && output % input == 0;
}

static bool is_fast_nearest_case(const resample_params& params) {
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    if (params.resampleType != ResampleType::NEAREST_NEIGHBOR || has_padding(params) ||
        input.Batch().v != output.Batch().v || input.Feature().v != output.Feature().v) {
        return false;
    }

    const auto asymmetric_floor = params.coordTransMode == CoordinateTransformationMode::ASYMMETRIC &&
                                  params.nearestMode == NearestMode::FLOOR;
    const auto asymmetric_simple_upsampling =
        params.coordTransMode == CoordinateTransformationMode::ASYMMETRIC &&
        params.nearestMode == NearestMode::SIMPLE &&
        is_integral_upsampling_ratio(output.X().v, input.X().v) &&
        is_integral_upsampling_ratio(output.Y().v, input.Y().v) &&
        (input.Dimentions() != 5 || is_integral_upsampling_ratio(output.Z().v, input.Z().v));
    const auto tf_half_pixel_for_nn_floor_upsampling =
        params.coordTransMode == CoordinateTransformationMode::TF_HALF_PIXEL_FOR_NN &&
        params.nearestMode == NearestMode::FLOOR &&
        is_integral_upsampling_ratio(output.X().v, input.X().v) &&
        is_integral_upsampling_ratio(output.Y().v, input.Y().v) &&
        (input.Dimentions() != 5 || is_integral_upsampling_ratio(output.Z().v, input.Z().v));
    const auto half_pixel_round_prefer_floor =
        params.coordTransMode == CoordinateTransformationMode::HALF_PIXEL &&
        params.nearestMode == NearestMode::ROUND_PREFER_FLOOR &&
        is_integral_ratio(output.X().v, input.X().v) &&
        is_integral_ratio(output.Y().v, input.Y().v) &&
        (input.Dimentions() != 5 || is_integral_ratio(output.Z().v, input.Z().v));

    return asymmetric_floor || asymmetric_simple_upsampling || tf_half_pixel_for_nn_floor_upsampling ||
           half_pixel_round_prefer_floor;
}

static bool is_fast_linear_onnx_case(const resample_params& params) {
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    if (params.resampleType != ResampleType::LINEAR_ONNX || has_padding(params) ||
        params.coordTransMode != CoordinateTransformationMode::HALF_PIXEL ||
        input.Batch().v != output.Batch().v || input.Feature().v != output.Feature().v) {
        return false;
    }

    if (!is_integral_upsampling_ratio(output.X().v, input.X().v) ||
        !is_integral_upsampling_ratio(output.Y().v, input.Y().v)) {
        return false;
    }

    return input.Dimentions() != 5 || is_integral_upsampling_ratio(output.Z().v, input.Z().v);
}

static bool is_fast_caffe_bilinear_interp_case(const resample_params& params) {
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    return params.resampleType == ResampleType::CAFFE_BILINEAR_INTERP &&
           !has_padding(params) &&
           input.Batch().v == output.Batch().v &&
           input.Feature().v == output.Feature().v;
}

static int get_axis_index(InterpolateAxis axis) {
    switch (axis) {
    case InterpolateAxis::BATCH:
        return 0;
    case InterpolateAxis::FEATURE:
        return 1;
    case InterpolateAxis::Z:
        return 2;
    case InterpolateAxis::Y:
        return 3;
    case InterpolateAxis::X:
        return 4;
    default:
        return 0;
    }
}

static std::vector<float> get_legacy_scales(const resample_params& params) {
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];
    auto pads_begin = params.pads_begin;
    auto pads_end = params.pads_end;
    if (pads_begin.size() == 4)
        pads_begin.insert(pads_begin.begin() + 2, 0);
    if (pads_end.size() == 4)
        pads_end.insert(pads_end.begin() + 2, 0);

    const auto b_size_padded = pads_begin[0] + input.Batch().v + pads_end[0];
    const auto f_size_padded = pads_begin[1] + input.Feature().v + pads_end[1];
    const auto z_size_padded = pads_begin[2] + input.Z().v + pads_end[2];
    const auto y_size_padded = pads_begin[3] + input.Y().v + pads_end[3];
    const auto x_size_padded = pads_begin[4] + input.X().v + pads_end[4];

    std::vector<float> scales = {
        static_cast<float>(b_size_padded) / static_cast<float>(output.Batch().v),
        static_cast<float>(f_size_padded) / static_cast<float>(output.Feature().v),
        static_cast<float>(z_size_padded) / static_cast<float>(output.Z().v),
        static_cast<float>(y_size_padded) / static_cast<float>(output.Y().v),
        static_cast<float>(x_size_padded) / static_cast<float>(output.X().v),
    };

    for (std::size_t i = 0; i < params.axes.size(); i++) {
        const int idx = get_axis_index(params.axes[i]);
        if (params.shapeCalculationMode == kernel_selector::ShapeCalculationMode::SCALES)
            scales[idx] = 1.f / params.scales[i];
    }

    return scales;
}

JitConstants ResampleKernelRef::GetJitConstants(const resample_params& params) const {
    JitConstants jit = ResampleKernelBase::GetJitConstants(params);

    if (is_fast_nearest_case(params)) {
        jit.RemoveConstant("SCALES");
        jit.AddConstant(MakeJitConstant("SCALES", get_legacy_scales(params)));
        jit.AddConstant(MakeJitConstant("RESAMPLE_FAST_NEAREST", 1));
    }

    if (is_fast_linear_onnx_case(params)) {
        jit.RemoveConstant("SCALES");
        jit.AddConstant(MakeJitConstant("SCALES", get_legacy_scales(params)));
        jit.AddConstant(MakeJitConstant("RESAMPLE_USE_LEGACY_SCALE", 1));
    }

    if (is_fast_caffe_bilinear_interp_case(params)) {
        jit.RemoveConstant("SCALES");
        jit.AddConstant(MakeJitConstant("SCALES", get_legacy_scales(params)));
        jit.AddConstant(MakeJitConstant("RESAMPLE_USE_LEGACY_SCALE", 1));
    }

    if (use_packing(params)) {
        jit.AddConstant(MakeJitConstant("PACK_SIZE", packing_factor(params)));
        jit.AddConstant(MakeJitConstant("FEATURE_PACKED_MODE", "1"));
    }

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 4) {
            idx_order = {"batch", "OF_ID", "oy", "ox"};
        } else if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5) {
            idx_order = {"batch", "OF_ID", "oz", "oy", "ox"};
        }

        FusedOpsConfiguration conf = {"", idx_order, "interp_val", GetAccumulatorType(params), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

ResampleKernelBase::DispatchData ResampleKernelRef::SetDefault(const resample_params& arg) const {
    auto dispatchData = Parent::SetDefault(arg);
    auto in_layout = arg.inputs[0].GetLayout();
    auto out_layout = arg.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X },
                                                                     { Tensor::DataChannelName::Y, Tensor::DataChannelName::Z },
                                                                     { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};

    if (use_packing(arg)) {
        auto pack = packing_factor(arg);
        dispatchData.gws = { arg.outputs[0].X().v, arg.outputs[0].Y().v * arg.outputs[0].Z().v,
                             CeilDiv(arg.outputs[0].Feature().v, pack) * arg.outputs[0].Batch().v };
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo, in_layout, out_layout, dims_by_gws);
    }

    return dispatchData;
}

KernelsPriority ResampleKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_7;
}
}  // namespace kernel_selector
