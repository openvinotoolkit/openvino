// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_kernel_opt.h"

#include <algorithm>
#include <vector>
#include <kernel_selector_utils.h>

namespace kernel_selector {

static constexpr size_t sub_group_size = 16;

size_t ResampleKernelOpt::GetOptimalBlockSize(const resample_params& params) const {
    std::vector<size_t> block_width = { 16, 8, 4, 2, 1 };
    for (auto& w : block_width)
        if (params.outputs[0].X().v % w == 0)
            return w;
    return 1;
}

static size_t GetOptimalDivisor(const size_t input_size, size_t max_val = 16) {
    for (size_t s = max_val; s > 0; --s) {
        if (input_size % s == 0) {
            return s;
        }
    }
    return 1;
}

ParamsKey ResampleKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);

    // 5d formats
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv32_fsv32);

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableResampleType(ResampleType::BILINEAR_INTERP);
    k.EnableResampleType(ResampleType::NEAREST_NEIGHBOR);
    k.EnableResampleType(ResampleType::CAFFE_BILINEAR_INTERP);
    return k;
}

DeviceFeaturesKey ResampleKernelOpt::get_required_device_features_key(const Params& params) const {
    return get_common_subgroups_device_features_key(params);
}

static size_t get_vec_size(const resample_params &params) {
    if (params.inputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32) {
        return 2;
    } else {
        return 1;
    }
}

static int get_feature_slice_size(const resample_params &params) {
    return static_cast<int>(16 * get_vec_size(params));
}

static bool is_integral_ratio(size_t lhs, size_t rhs) {
    return lhs != 0 && rhs != 0 && (lhs % rhs == 0 || rhs % lhs == 0);
}

static bool is_integral_upsampling_ratio(size_t output, size_t input) {
    return input != 0 && output >= input && output % input == 0;
}

static bool is_asymmetric_simple_optimized_case(const resample_params& params) {
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    if (params.coordTransMode != CoordinateTransformationMode::ASYMMETRIC || params.nearestMode != NearestMode::SIMPLE) {
        return false;
    }

    if (!is_integral_upsampling_ratio(output.X().v, input.X().v) ||
        !is_integral_upsampling_ratio(output.Y().v, input.Y().v)) {
        return false;
    }

    return input.Dimentions() != 5 || is_integral_upsampling_ratio(output.Z().v, input.Z().v);
}

static bool is_tf_half_pixel_for_nn_floor_optimized_case(const resample_params& params) {
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    if (params.coordTransMode != CoordinateTransformationMode::TF_HALF_PIXEL_FOR_NN ||
        params.nearestMode != NearestMode::FLOOR) {
        return false;
    }

    if (!is_integral_upsampling_ratio(output.X().v, input.X().v) ||
        !is_integral_upsampling_ratio(output.Y().v, input.Y().v)) {
        return false;
    }

    return input.Dimentions() != 5 || is_integral_upsampling_ratio(output.Z().v, input.Z().v);
}

static bool is_half_pixel_round_prefer_floor_optimized_case(const resample_params& params) {
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    if (params.coordTransMode != CoordinateTransformationMode::HALF_PIXEL ||
        params.nearestMode != NearestMode::ROUND_PREFER_FLOOR) {
        return false;
    }

    if (!is_integral_ratio(output.X().v, input.X().v) || !is_integral_ratio(output.Y().v, input.Y().v)) {
        return false;
    }

    return input.Dimentions() != 5 || is_integral_ratio(output.Z().v, input.Z().v);
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

ResampleKernelBase::DispatchData ResampleKernelOpt::SetDefault(const kernel_selector::resample_params &arg) const {
    DispatchData dispatchData;
    auto in_layout = arg.inputs[0].GetLayout();
    auto out_layout = arg.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    size_t dims = arg.outputs[0].Dimentions();
    const auto& out = arg.outputs[0];

    if (arg.resampleType == ResampleType::CAFFE_BILINEAR_INTERP) {
        dispatchData.gws[0] = out.X().v * out.Y().v;
        dispatchData.gws[1] = CeilDiv(out.Feature().v, GetFeatureBlockSize(arg));
        dispatchData.gws[2] = arg.outputs[0].Batch().v;

        dims_by_gws = {{Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::FEATURE},
                       {Tensor::DataChannelName::BATCH}};
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo, in_layout, out_layout, dims_by_gws);
    } else {
        auto opt_x_block_size = GetOptimalBlockSize(arg);
        if (out.X().v > 32 && opt_x_block_size == 1) {
            opt_x_block_size = GetOptimalDivisor(out.X().v, 32);
        }

        if (dims == 5) {
            dispatchData.gws[0] = CeilDiv(out.X().v, opt_x_block_size) * out.Y().v * out.Z().v;
        } else {
            dispatchData.gws[0] = CeilDiv(out.X().v, opt_x_block_size) * out.Y().v;
        }
        dispatchData.gws[1] = Align(CeilDiv(out.Feature().v, get_vec_size(arg)), sub_group_size);
        dispatchData.gws[2] = arg.outputs[0].Batch().v;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = sub_group_size;
        dispatchData.lws[2] = 1;

        if (arg.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16 ||
            arg.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv32 ||
            arg.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16 ||
            arg.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv32 ||
            arg.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv16 ||
            arg.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv32) {
            dispatchData.lws[2] = GetOptimalDivisor(dispatchData.gws[2]);
        }
    }

    return dispatchData;
}

KernelsPriority ResampleKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}

bool ResampleKernelOpt::Validate(const Params& p) const {
    const resample_params& params = static_cast<const resample_params&>(p);
    if (!Parent::Validate(p))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    const auto has_padding = std::any_of(params.pads_begin.begin(), params.pads_begin.end(), [](const auto pad) { return pad != 0; }) ||
                             std::any_of(params.pads_end.begin(), params.pads_end.end(), [](const auto pad) { return pad != 0; });

    if ((input.GetDType() == Datatype::UINT8 || input.GetDType() == Datatype::INT8) &&
        params.resampleType != ResampleType::NEAREST_NEIGHBOR &&
        params.resampleType != ResampleType::BILINEAR_INTERP)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // in the case of 5D support only NEAREST_NEIGHBOR
    if (input.Dimentions() == 5 && params.resampleType != ResampleType::NEAREST_NEIGHBOR)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.resampleType == ResampleType::NEAREST_NEIGHBOR) {
        const auto optimized_nearest_case =
            (params.coordTransMode == CoordinateTransformationMode::ASYMMETRIC && params.nearestMode == NearestMode::FLOOR) ||
            is_asymmetric_simple_optimized_case(params) ||
            is_tf_half_pixel_for_nn_floor_optimized_case(params) ||
            is_half_pixel_round_prefer_floor_optimized_case(params);

        if (has_padding ||
            !optimized_nearest_case ||
            input.Batch().v != output.Batch().v ||
            input.Feature().v != output.Feature().v) {
            DO_NOT_USE_THIS_KERNEL(p.layerID);
        }
    }

    if (params.resampleType == ResampleType::BILINEAR_INTERP) {
        if (has_padding ||
            params.coordTransMode != CoordinateTransformationMode::ASYMMETRIC ||
            input.Batch().v != output.Batch().v ||
            input.Feature().v != output.Feature().v) {
            DO_NOT_USE_THIS_KERNEL(p.layerID);
        }
    }

    return true;
}

JitConstants ResampleKernelOpt::GetJitConstants(const resample_params &params) const {
    auto jit = Parent::GetJitConstants(params);
    jit.RemoveConstant("SCALES");
    jit.AddConstant(MakeJitConstant("SCALES", get_legacy_scales(params)));

    auto opt_x_block_size = GetOptimalBlockSize(params);
    if (params.outputs[0].X().v > 32 && opt_x_block_size == 1) {
        opt_x_block_size = GetOptimalDivisor(params.outputs[0].X().v, 32);
    }

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", opt_x_block_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.outputs[0].X().v, opt_x_block_size)));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));

    const size_t vec_size = get_vec_size(params);
    jit.AddConstant(MakeJitConstant("FEATURE_SLICE_SIZE", get_feature_slice_size(params)));
    jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));

    if (!params.fused_ops.empty()) {
        if (params.resampleType != ResampleType::CAFFE_BILINEAR_INTERP) {
            std::vector<std::string> idx_order;
            if (params.inputs[0].Dimentions() == 5)
                idx_order = {"b", "feature_block", "z", "y", "(x + out_x)"};
            else
                idx_order = {"b", "feature_block", "y", "(x + out_x)"};
            FusedOpsConfiguration conf = {"", idx_order, "res", GetAccumulatorType(params), vec_size, LoadType::LT_ALIGNED_READ};
            conf.SetVectorAxis(Tensor::DataChannelName::FEATURE);
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        } else {
            std::vector<std::string> idx_order;
            idx_order = {"batch", "OF_ID", "oy", "ox"};

            FusedOpsConfiguration conf = {"", idx_order, "res", GetAccumulatorType(params), 1};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        }
    }

    if (params.resampleType == ResampleType::CAFFE_BILINEAR_INTERP) {
        if (GetFeatureBlockSize(params) == 8) {
            jit.AddConstant(MakeJitConstant("VEC_BLOCK_SIZE", 8));
        } else {
            jit.AddConstant(MakeJitConstant("VEC_BLOCK_SIZE", 16));
        }
    }

    return jit;
}

Datatype ResampleKernelOpt::GetUnitType(const base_params& params) const {
    return params.inputs[0].GetDType();
}

KernelsData ResampleKernelOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}
}  // namespace kernel_selector
