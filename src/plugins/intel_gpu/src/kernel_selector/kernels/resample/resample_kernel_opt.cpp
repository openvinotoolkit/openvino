// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_kernel_opt.h"

#include <algorithm>
#include <vector>
#include <kernel_selector_utils.h>
#include "resample/utils.hpp"

namespace kernel_selector {

static constexpr size_t sub_group_size = 16;

static size_t get_vec_size(const resample_params& params);
static size_t get_feature_slices_per_wi(const resample_params& params);

size_t ResampleKernelOpt::GetOptimalBlockSize(const resample_params& params) const {
    size_t max_width = 16;
    if (params.outputs[0].GetLayout() == DataLayout::byxf) {
        // A work-item that covers whole pixels writes contiguous memory: about 1 KiB per work-item is the fastest.
        // When several work-items share a pixel, 4 pixels per work-item is the fastest.
        const auto& out = params.outputs[0];
        const size_t wi_features = sub_group_size * get_vec_size(params) * get_feature_slices_per_wi(params);
        const size_t pixel_bytes = out.Feature().v * BytesPerElement(out.GetDType());
        max_width = wi_features == out.Feature().v ? std::clamp<size_t>(1024 / pixel_bytes, 4, 16) : 4;
    }
    std::vector<size_t> block_width = { 16, 8, 4, 2, 1 };
    for (auto& w : block_width) {
        if (w <= max_width && params.outputs[0].X().v % w == 0) {
            return w;
        }
    }
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
    k.EnableInputDataType(Datatype::BF16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::BF16);
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
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::byxf);

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
    k.EnableResampleType(ResampleType::LINEAR_ONNX);
    k.EnableResampleType(ResampleType::NEAREST_NEIGHBOR);
    k.EnableResampleType(ResampleType::CAFFE_BILINEAR_INTERP);
    return k;
}

DeviceFeaturesKey ResampleKernelOpt::get_required_device_features_key(const Params& params) const {
    return get_common_subgroups_device_features_key(params);
}

static bool is_byxf(const resample_params& params) {
    return params.inputs[0].GetLayout() == DataLayout::byxf || params.outputs[0].GetLayout() == DataLayout::byxf;
}

// Nearest asymmetric/floor upsampling by integer factors: output (y, x) reads input (y / UPSAMPLE_Y, x / UPSAMPLE_X).
// The output x-block must hold whole groups of UPSAMPLE_X outputs, so that each input pixel is read once.
static bool is_byxf_integral_nearest(const resample_params& params, size_t x_block_size) {
    const auto& in = params.inputs[0];
    const auto& out = params.outputs[0];
    if (!is_byxf(params) || params.resampleType != ResampleType::NEAREST_NEIGHBOR || params.coordTransMode != CoordinateTransformationMode::ASYMMETRIC ||
        params.nearestMode != NearestMode::FLOOR || ResampleKernelBase::has_padding(params) || in.Batch().v != out.Batch().v ||
        in.Feature().v != out.Feature().v || !is_integral_upsampling_ratio(out.X().v, in.X().v) || !is_integral_upsampling_ratio(out.Y().v, in.Y().v) ||
        x_block_size % (out.X().v / in.X().v) != 0) {
        return false;
    }
    // With SCALES the kernel maps by the scale value, which can differ from the size ratio (scale 2.2 on X = 4 gives 8).
    if (params.shapeCalculationMode == ShapeCalculationMode::SCALES) {
        for (size_t i = 0; i < params.axes.size(); ++i) {
            if ((params.axes[i] == InterpolateAxis::X && params.scales[i] != static_cast<float>(out.X().v / in.X().v)) ||
                (params.axes[i] == InterpolateAxis::Y && params.scales[i] != static_cast<float>(out.Y().v / in.Y().v))) {
                return false;
            }
        }
    }
    return true;
}

static size_t get_vec_size(const resample_params &params) {
    if (params.inputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32) {
        return 2;
    }
    if (is_byxf(params)) {
        // Features of a pixel are contiguous: use the widest sub-group block IO that divides F.
        for (size_t vec = 8; vec > 1; vec /= 2) {
            if (params.outputs[0].Feature().v % (sub_group_size * vec) == 0) {
                return vec;
            }
        }
    }
    return 1;
}

static int get_feature_slice_size(const resample_params &params) {
    return static_cast<int>(16 * get_vec_size(params));
}

// When a sub-group writes at most one 64 B cache line per byxf pixel (16 * VEC_SIZE elements), one work-item handles
// all feature slices of the pixel. Otherwise several work-items write parts of the same cache lines.
static size_t get_feature_slices_per_wi(const resample_params& params) {
    const auto& out = params.outputs[0];
    if (!is_byxf(params) || sub_group_size * get_vec_size(params) * BytesPerElement(out.GetDType()) > 64) {
        return 1;
    }
    return out.Feature().v / (sub_group_size * get_vec_size(params));
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

        if (is_byxf(arg)) {
            // Features fastest: neighboring sub-groups access adjacent parts of the same pixels.
            // The fsv16 mapping below would walk pixels at stride F and is up to 13x slower for byxf.
            const size_t features_per_wi = get_vec_size(arg) * get_feature_slices_per_wi(arg);
            dispatchData.gws = {out.Feature().v / features_per_wi, CeilDiv(out.X().v, opt_x_block_size) * out.Y().v, out.Batch().v};
            size_t lws0 = 256;
            while (lws0 > sub_group_size && (dispatchData.gws[0] % lws0 != 0 || lws0 > arg.engineInfo.maxWorkGroupSize)) {
                lws0 /= 2;
            }
            dispatchData.lws = {lws0, 1, 1};
            return dispatchData;
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
    if (!Parent::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    const auto has_padding = ResampleKernelBase::has_padding(params);

    if (is_byxf(params)) {
        // A sub-group block access covers 16 * VEC_SIZE consecutive features of one pixel.
        const bool feature_padded = input.Feature().pad.Total() != 0 || output.Feature().pad.Total() != 0;
        const bool supported_type = params.resampleType == ResampleType::NEAREST_NEIGHBOR || params.resampleType == ResampleType::LINEAR_ONNX;
        if (input.GetLayout() != output.GetLayout() || input.Feature().v % sub_group_size != 0 || feature_padded || !supported_type) {
            DO_NOT_USE_THIS_KERNEL(p.layerID);
        }
        // Fused-op aligned loads use the data lane/feature mapping, which only holds for byxf or broadcast inputs.
        for (const auto& fused_op : params.fused_ops) {
            for (const auto& t : fused_op.tensors) {
                const bool scalar = t.LogicalSize() == 1;
                const bool per_channel = t.Feature().v == output.Feature().v && t.LogicalSize() == output.Feature().v;
                const bool byxf_tensor = t.GetLayout() == DataLayout::byxf && t.Feature().v == output.Feature().v;
                if (!scalar && !per_channel && !byxf_tensor) {
                    DO_NOT_USE_THIS_KERNEL(p.layerID);
                }
            }
        }
    }

    if ((input.GetDType() == Datatype::UINT8 || input.GetDType() == Datatype::INT8) &&
        params.resampleType != ResampleType::NEAREST_NEIGHBOR &&
        params.resampleType != ResampleType::BILINEAR_INTERP &&
        params.resampleType != ResampleType::LINEAR_ONNX) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    // in the case of 5D support only NEAREST_NEIGHBOR
    if (input.Dimentions() == 5 && params.resampleType != ResampleType::NEAREST_NEIGHBOR) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    // LINEAR_ONNX in resample_opt does not implement OOB-to-zero for padded
    // coordinates; let resample_onnx (which does) handle those cases.
    if (params.resampleType == ResampleType::LINEAR_ONNX) {
        if (input.Batch().v != output.Batch().v || input.Feature().v != output.Feature().v) {
            DO_NOT_USE_THIS_KERNEL(p.layerID);
        }

        for (size_t i = 0; i < params.pads_begin.size(); ++i) {
            if (params.pads_begin[i] != 0 || params.pads_end[i] != 0) {
                DO_NOT_USE_THIS_KERNEL(p.layerID);
            }
        }
    }

    if (params.resampleType == ResampleType::NEAREST_NEIGHBOR) {
        // The NEAREST kernel handles every CoordinateTransformationMode/NearestMode combination
        // via get_original_coordinate()/get_nearest_val(); padding and batch/feature resize are not supported.
        if (has_padding ||
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

JitConstants ResampleKernelOpt::GetJitConstants(const resample_params &params, bool legacy_scale) const {
    auto jit = Parent::GetJitConstants(params, true);

    auto opt_x_block_size = GetOptimalBlockSize(params);
    if (params.outputs[0].X().v > 32 && opt_x_block_size == 1) {
        opt_x_block_size = GetOptimalDivisor(params.outputs[0].X().v, 32);
    }

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", opt_x_block_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.outputs[0].X().v, opt_x_block_size)));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("FEATURE_SLICES_PER_WI", get_feature_slices_per_wi(params)));
    if (is_byxf_integral_nearest(params, opt_x_block_size)) {
        jit.AddConstant(MakeJitConstant("BYXF_INTEGRAL_NEAREST", 1));
        jit.AddConstant(MakeJitConstant("UPSAMPLE_X", params.outputs[0].X().v / params.inputs[0].X().v));
        jit.AddConstant(MakeJitConstant("UPSAMPLE_Y", params.outputs[0].Y().v / params.inputs[0].Y().v));
    }

    const size_t vec_size = get_vec_size(params);
    jit.AddConstant(MakeJitConstant("FEATURE_SLICE_SIZE", get_feature_slice_size(params)));
    jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));

    if (!params.fused_ops.empty()) {
        if (params.resampleType != ResampleType::CAFFE_BILINEAR_INTERP) {
            std::vector<std::string> idx_order;
            if (params.inputs[0].Dimentions() == 5) {
                idx_order = {"b", "feature_block", "z", "y", "(x + out_x)"};
            } else {
                idx_order = {"b", "feature_block", "y", "(x + out_x)"};
            }
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
