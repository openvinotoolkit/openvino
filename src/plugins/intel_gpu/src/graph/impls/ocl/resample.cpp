// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "resample_inst.h"
#include "kernel_selector/kernels/resample/resample_kernel_selector.h"
#include "kernel_selector/kernels/resample/resample_kernel_base.h"
#include <set>

namespace cldnn {
namespace ocl {

namespace {
inline kernel_selector::sample_type convert_to_sample_type(resample::InterpolateOp::InterpolateMode type) {
    switch (type) {
        case resample::InterpolateOp::InterpolateMode::NEAREST:
            return kernel_selector::sample_type::NEAREST_NEIGHBOR;
        case resample::InterpolateOp::InterpolateMode::LINEAR:
            return kernel_selector::sample_type::CAFFE_BILINEAR_INTERP;
        case resample::InterpolateOp::InterpolateMode::CUBIC:
            return kernel_selector::sample_type::CUBIC;
        case resample::InterpolateOp::InterpolateMode::LINEAR_ONNX:
            return kernel_selector::sample_type::LINEAR_ONNX;
        case resample::InterpolateOp::InterpolateMode::BILINEAR_PILLOW:
            return kernel_selector::sample_type::BILINEAR_PILLOW;
        case resample::InterpolateOp::InterpolateMode::BICUBIC_PILLOW:
            return kernel_selector::sample_type::BICUBIC_PILLOW;
        default:
            return kernel_selector::sample_type::NEAREST_NEIGHBOR;
    }
}

inline kernel_selector::coordinate_transformation_mode convert_to_coord_transform_mode(resample::InterpolateOp::CoordinateTransformMode mode) {
    switch (mode) {
        case resample::InterpolateOp::CoordinateTransformMode::HALF_PIXEL:
            return kernel_selector::coordinate_transformation_mode::HALF_PIXEL;
        case resample::InterpolateOp::CoordinateTransformMode::PYTORCH_HALF_PIXEL:
            return kernel_selector::coordinate_transformation_mode::PYTORCH_HALF_PIXEL;
        case resample::InterpolateOp::CoordinateTransformMode::ASYMMETRIC:
            return kernel_selector::coordinate_transformation_mode::ASYMMETRIC;
        case resample::InterpolateOp::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN:
            return kernel_selector::coordinate_transformation_mode::TF_HALF_PIXEL_FOR_NN;
        case resample::InterpolateOp::CoordinateTransformMode::ALIGN_CORNERS:
            return kernel_selector::coordinate_transformation_mode::ALIGN_CORNERS;
        default:
            return kernel_selector::coordinate_transformation_mode::HALF_PIXEL;
    }
}

inline kernel_selector::nearest_mode convert_to_nearest_mode(resample::InterpolateOp::NearestMode mode) {
    switch (mode) {
        case resample::InterpolateOp::NearestMode::ROUND_PREFER_FLOOR:
            return kernel_selector::nearest_mode::ROUND_PREFER_FLOOR;
        case resample::InterpolateOp::NearestMode::ROUND_PREFER_CEIL:
            return kernel_selector::nearest_mode::ROUND_PREFER_CEIL;
        case resample::InterpolateOp::NearestMode::FLOOR:
            return kernel_selector::nearest_mode::FLOOR;
        case resample::InterpolateOp::NearestMode::CEIL:
            return kernel_selector::nearest_mode::CEIL;
        case resample::InterpolateOp::NearestMode::SIMPLE:
            return kernel_selector::nearest_mode::SIMPLE;
        default:
            return kernel_selector::nearest_mode::ROUND_PREFER_FLOOR;
    }
}

inline kernel_selector::shape_calculation_mode convert_to_shape_calculation_mode(resample::InterpolateOp::ShapeCalcMode mode) {
    switch (mode) {
        case resample::InterpolateOp::ShapeCalcMode::SIZES:
            return kernel_selector::shape_calculation_mode::SIZES;
        case resample::InterpolateOp::ShapeCalcMode::SCALES:
            return kernel_selector::shape_calculation_mode::SCALES;
        default:
            return kernel_selector::shape_calculation_mode::SIZES;
    }
}

inline std::vector<int32_t> convert_pads(const std::vector<size_t>& pad, size_t rank) {
    std::vector<int32_t> new_pad;

    if (pad.empty()) {
        new_pad = std::vector<int32_t>(rank, 0);
    } else {
        for (auto p : pad) {
            new_pad.push_back(static_cast<int32_t>(p));
        }
        if (new_pad.size() > 2)
            std::reverse(new_pad.begin() + 2, new_pad.end());
        for (size_t i = new_pad.size(); i < rank || i < 4; ++i)
            new_pad.push_back(0);
    }

    return new_pad;
}

inline kernel_selector::interpolate_axis convert_axis(int64_t axis, size_t rank) {
    switch (axis) {
        case 0:
            return kernel_selector::interpolate_axis::BATCH;
        case 1:
            return kernel_selector::interpolate_axis::FEATURE;
        case 2:
            if (rank == 6)
                return kernel_selector::interpolate_axis::W;
            else if (rank == 5)
                return kernel_selector::interpolate_axis::Z;
            else
                return kernel_selector::interpolate_axis::Y;
        case 3:
            if (rank == 6)
                return kernel_selector::interpolate_axis::Z;
            else if (rank == 5)
                return kernel_selector::interpolate_axis::Y;
            else
                return kernel_selector::interpolate_axis::X;
        case 4:
            if (rank == 6)
                return kernel_selector::interpolate_axis::Y;
            else
                return kernel_selector::interpolate_axis::X;
        case 5:
            return kernel_selector::interpolate_axis::X;
        default:
            throw std::runtime_error("Unsupported axis for interpolate (" + std::to_string(axis) + ")");
    }
}
}  // namespace

struct resample_impl : typed_primitive_impl_ocl<resample> {
    using parent = typed_primitive_impl_ocl<resample>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::resample_kernel_selector;
    using kernel_params_t = kernel_selector::resample_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::resample_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<resample_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<resample>();
        auto params = get_default_params<kernel_selector::resample_params>(impl_param);

        size_t dimsNum = impl_param.get_output_layout().get_rank();
        params.resampleType = convert_to_sample_type(primitive->operation_type);
        params.nearestMode = convert_to_nearest_mode(primitive->round_mode);
        params.coordTransMode = convert_to_coord_transform_mode(primitive->coord_trans_mode);
        params.shapeCalculationMode = convert_to_shape_calculation_mode(primitive->shape_calc_mode);
        params.antialias = primitive->antialias;
        params.cube_coeff = primitive->cube_coeff;

        params.pads_begin = convert_pads(primitive->pads_begin, dimsNum);
        params.pads_end = convert_pads(primitive->pads_end, dimsNum);

        auto scales = primitive->scales;
        auto scales_port = primitive->scales_port;
        bool scales_calc_mod = primitive->shape_calc_mode == resample::InterpolateOp::ShapeCalcMode::SCALES;
        if (scales_calc_mod && impl_param.input_layouts.size() > 1 && scales.empty()) {
            auto mem = impl_param.memory_deps.at(scales_port);
            scales = read_vector<float>(std::move(mem), impl_param.get_stream());
        }

        params.scales = scales;
        std::vector<kernel_selector::InterpolateAxis> axes;
        std::transform(primitive->axes.begin(), primitive->axes.end(),
                       std::back_inserter(axes),
                       [dimsNum](std::int64_t axis){ return convert_axis(axis, dimsNum); });
        params.axes = std::move(axes);

        return params;
    }
};

namespace detail {

attach_resample_impl::attach_resample_impl() {
    std::set<implementation_map<resample>::key_type> keys;

    const auto types = {data_types::f16, data_types::f32, data_types::i8, data_types::u8, data_types::i32};
    const auto formats = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,

        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,
    };
    for (const auto type : types) {
        for (const auto format : formats) {
            keys.emplace(type, format);
        }
    }

    keys.emplace(data_types::f32, format::yxfb);
    keys.emplace(data_types::f16, format::yxfb);
    keys.emplace(data_types::f16, format::fs_b_yx_fsv32);

    implementation_map<resample>::add(impl_types::ocl, typed_primitive_impl_ocl<resample>::create<resample_impl>, keys);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::resample_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::resample)
