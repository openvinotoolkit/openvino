// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "kernel_selector/core/actual_kernels/resample/resample_kernel_selector.h"
#include "kernel_selector/core/actual_kernels/resample/resample_kernel_base.h"

namespace cldnn {
namespace ocl {

namespace {
inline kernel_selector::sample_type convert_to_sample_type(resample_type type) {
    switch (type) {
        case resample_type::nearest:
            return kernel_selector::sample_type::NEAREST_NEIGHBOR;
        case resample_type::caffe_bilinear:
            return kernel_selector::sample_type::CAFFE_BILINEAR_INTERP;
        case resample_type::bilinear:
            return kernel_selector::sample_type::BILINEAR_INTERP;
        case resample_type::cubic:
            return kernel_selector::sample_type::CUBIC;
        case resample_type::linear_onnx:
            return kernel_selector::sample_type::LINEAR_ONNX;
        default:
            return kernel_selector::sample_type::NEAREST_NEIGHBOR;
    }
}

inline kernel_selector::coordinate_transformation_mode convert_to_coord_transform_mode(coordinate_transformation_mode mode) {
    switch (mode) {
        case coordinate_transformation_mode::half_pixel:
            return kernel_selector::coordinate_transformation_mode::HALF_PIXEL;
        case coordinate_transformation_mode::pytorch_half_pixel:
            return kernel_selector::coordinate_transformation_mode::PYTORCH_HALF_PIXEL;
        case coordinate_transformation_mode::asymmetric:
            return kernel_selector::coordinate_transformation_mode::ASYMMETRIC;
        case coordinate_transformation_mode::tf_half_pixel_for_nn:
            return kernel_selector::coordinate_transformation_mode::TF_HALF_PIXEL_FOR_NN;
        case coordinate_transformation_mode::align_corners:
            return kernel_selector::coordinate_transformation_mode::ALIGN_CORNERS;
        default:
            return kernel_selector::coordinate_transformation_mode::HALF_PIXEL;
    }
}

inline kernel_selector::nearest_mode convert_to_nearest_mode(nearest_mode mode) {
    switch (mode) {
        case nearest_mode::round_prefer_floor:
            return kernel_selector::nearest_mode::ROUND_PREFER_FLOOR;
        case nearest_mode::round_prefer_ceil:
            return kernel_selector::nearest_mode::ROUND_PREFER_CEIL;
        case nearest_mode::floor:
            return kernel_selector::nearest_mode::FLOOR;
        case nearest_mode::ceil:
            return kernel_selector::nearest_mode::CEIL;
        case nearest_mode::simple:
            return kernel_selector::nearest_mode::SIMPLE;
        default:
            return kernel_selector::nearest_mode::ROUND_PREFER_FLOOR;
    }
}

inline kernel_selector::shape_calculation_mode convert_to_shape_calculation_mode(shape_calculation_mode mode) {
    switch (mode) {
        case shape_calculation_mode::sizes:
            return kernel_selector::shape_calculation_mode::SIZES;
        case shape_calculation_mode::scales:
            return kernel_selector::shape_calculation_mode::SCALES;
        default:
            return kernel_selector::shape_calculation_mode::SIZES;
    }
}

inline kernel_selector::interpolate_axis convert_axis(resample::resample_axis axis) {
    switch (axis) {
        case resample::along_x:
            return kernel_selector::interpolate_axis::X;
        case resample::along_y:
            return kernel_selector::interpolate_axis::Y;
        case resample::along_z:
            return kernel_selector::interpolate_axis::Z;
        case resample::along_w:
            return kernel_selector::interpolate_axis::W;
        case resample::along_f:
            return kernel_selector::interpolate_axis::FEATURE;
        case resample::along_b:
            return kernel_selector::interpolate_axis::BATCH;
        default:
            return kernel_selector::interpolate_axis::BATCH;
    }
}
}  // namespace

struct resample_impl : typed_primitive_impl_ocl<resample> {
    using parent = typed_primitive_impl_ocl<resample>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<resample_impl>(*this);
    }

    static primitive_impl* create(const resample_node& arg) {
        auto us_params = get_default_params<kernel_selector::resample_params>(arg);
        auto us_optional_params =
            get_default_optional_params<kernel_selector::resample_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();
        size_t dimsNum = arg.get_output_layout().format.dimension();
        us_params.resampleType = convert_to_sample_type(primitive->operation_type);
        us_params.nearestMode = convert_to_nearest_mode(primitive->round_mode);
        us_params.coordTransMode = convert_to_coord_transform_mode(primitive->coord_trans_mode);
        us_params.shapeCalculationMode = convert_to_shape_calculation_mode(primitive->shape_calc_mode);
        us_params.antialias = primitive->antialias;
        us_params.cube_coeff = primitive->cube_coeff;
        us_params.pads_begin = primitive->pads_begin.empty() ? std::vector<int32_t>(dimsNum, 0) : primitive->pads_begin;
        us_params.pads_end = primitive->pads_end.empty() ? std::vector<int32_t>(dimsNum, 0) : primitive->pads_end;
        for (const auto& it : primitive->axesAndScales) {
            us_params.axesAndScales[convert_axis(it.first)] = it.second;
        }

        if (primitive->operation_type == resample_type::bilinear) {
            us_params.align_corners = primitive->align_corners;
        }

        auto& kernel_selector = kernel_selector::resample_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(us_params, us_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto resample = new resample_impl(arg, best_kernels[0]);

        return resample;
    }
};

namespace detail {

attach_resample_impl::attach_resample_impl() {
    implementation_map<resample>::add(impl_types::ocl, resample_impl::create, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),

        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),

        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),

        std::make_tuple(data_types::f16, format::fs_b_yx_fsv32),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
