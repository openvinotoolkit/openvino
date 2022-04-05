// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"

#include <map>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Sample mode for the @ref resample layer.
enum class resample_type : int32_t {
    /// @brief nearest neighbor.
    nearest,
    /// @brief bilinear interpolation.
    bilinear,
    /// @brief caffe bilinear interpolation.
    caffe_bilinear,
    /// @brief cubic interpolation.
    cubic,
    /// @brief linear onnx interpolation.
    linear_onnx
};

/// @brief Specifies which of inputs target_spatial_shape or scales is used to calculate an output shape
enum class shape_calculation_mode : int32_t {
    /// @brief output shape calculated based on sizes of input and output tensors
    sizes,
    /// @brief output shape calculated based on scales coefficients
    scales
};

/// @brief Coordinate transformation mode for the @ref resample layer.
enum class coordinate_transformation_mode : int32_t {
    /// @brief the coordinate in the original tensor axis `x` is calculated as `((x_resized + 0.5) / scale[x]) - 0.5`.
    half_pixel,
    /// @brief the coordinate in the original tensor axis `x` is calculated by `(x_resized + 0.5) / scale[x] - 0.5 if out_shape[x] > 1 else 0.0`.
    pytorch_half_pixel,
    /// @brief the coordinate in the original tensor axis `x` is calculated according to the formula `x_resized / scale[x]`.
    asymmetric,
    /// @brief the coordinate in the original tensor axis `x` is `(x_resized + 0.5) / scale[x]`.
    tf_half_pixel_for_nn,
    /// @brief the coordinate in the original tensor axis `x` is calculated as `0 if out_shape[x] == 1 else x_resized * (in_shape[x] - 1) / (out_shape[x] - 1)`.
    align_corners
};

/// @brief Nearest mode for the @ref resample layer.
enum class nearest_mode : int32_t {
    /// @brief this mode is known as round half down.
    round_prefer_floor,
    /// @brief it is round half up mode.
    round_prefer_ceil,
    /// @brief this mode computes the largest integer value not greater than rounded value.
    floor,
    /// @brief this mode computes the smallest integer value not less than rounded value
    ceil,
    /// @brief this mode behaves as `ceil` mode when `Interpolate` is downsample, and as dropping the fractional part otherwise.
    simple
};

/// @brief Performs nearest neighbor/bilinear resample
/// Also supports built-in Relu @ref activation available by setting it in arguments.
struct resample : public primitive_base<resample> {
    CLDNN_DECLARE_PRIMITIVE(resample)

    enum resample_axis {
        along_b,
        along_f,
        along_x,
        along_y,
        along_z,
        along_w
    };

    using AxesAndScales = std::map<resample_axis, float>;

    /// @brief Constructs Resample primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param scale Resample scale.
    /// @param num_filter Input filter. Only used by bilinear sample_type.
    /// @param sample_type Resample method (nearest neighbor/bilinear/caffe bilinear).
    resample(const primitive_id& id,
             const primitive_id& input,
             tensor output_size,
             uint32_t num_filter,
             resample_type operation_type = resample_type::nearest,
             const primitive_id& ext_prim_id = "",
             const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          output_size(output_size),
          num_filter(num_filter),
          axesAndScales({}),
          pads_begin({}),
          pads_end({}),
          align_corners(1),
          operation_type(operation_type),
          shape_calc_mode(shape_calculation_mode::sizes),
          antialias(0),
          cube_coeff(0.0f),
          coord_trans_mode(coordinate_transformation_mode::asymmetric),
          round_mode(nearest_mode::floor) {
        if (operation_type == resample_type::caffe_bilinear) {
            coord_trans_mode = coordinate_transformation_mode::half_pixel;
        }
    }

    /// @brief Constructs Resample primitive with Interp operation.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param pads_begin Optional begin padding for input.
    /// @param pads_end Optional end padding for input.
    /// @param align_corners Align corner pixels of the input and output tensors.
    /// @param resample_type Resample bilinear method.
    resample(const primitive_id& id,
             const primitive_id& input,
             tensor output_size,
             std::vector<int32_t> pads_begin = {},
             std::vector<int32_t> pads_end = {},
             int32_t align_corners = 1,
             resample_type operation_type = resample_type::bilinear,
             const primitive_id& ext_prim_id = "",
             const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          output_size(output_size),
          num_filter(0),
          axesAndScales({}),
          pads_begin(pads_begin),
          pads_end(pads_end),
          align_corners(align_corners),
          operation_type(operation_type),
          shape_calc_mode(shape_calculation_mode::sizes),
          antialias(0),
          cube_coeff(0.0f),
          coord_trans_mode(coordinate_transformation_mode::asymmetric),
          round_mode(nearest_mode::floor) {}

    /// @brief Constructs Resample primitive with Interpolate operation.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param pads_begin Optional begin padding for input.
    /// @param pads_end Optional end padding for input.
    resample(const primitive_id& id,
             const primitive_id& input,
             tensor output_size,
             AxesAndScales axesAndScales,
             std::vector<int32_t> pads_begin = {},
             std::vector<int32_t> pads_end = {},
             int32_t antialias = 0,
             float cube_coeff = -0.75f,
             resample_type operation_type = resample_type::caffe_bilinear,
             shape_calculation_mode shape_calc_mode = shape_calculation_mode::sizes,
             coordinate_transformation_mode ctm = coordinate_transformation_mode::half_pixel,
             nearest_mode nm = nearest_mode::round_prefer_floor,
             const primitive_id& ext_prim_id = "",
             const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          output_size(output_size),
          num_filter(0),
          axesAndScales(axesAndScales),
          pads_begin(pads_begin),
          pads_end(pads_end),
          align_corners(1),
          operation_type(operation_type),
          shape_calc_mode(shape_calc_mode),
          antialias(antialias),
          cube_coeff(cube_coeff),
          coord_trans_mode(ctm),
          round_mode(nm) {}

    /// @param scale Resample scale.
    tensor output_size;
    /// @param num_filter Input filter. Only used by bilinear sample_type.
    uint32_t num_filter;
    /// @param scales scales for spatial axes.
    AxesAndScales axesAndScales;
    /// @param pads_begin Begin paddings for input.
    std::vector<int32_t> pads_begin;
    /// @param pads_end End paddings for input.
    std::vector<int32_t> pads_end;
    /// @param align_corners corner pixels of the input and output tensors
    int32_t align_corners;
    /// @param operation_type Resample method (nearest neighbor/bilinear/caffe bilinear).
    resample_type operation_type;
    /// @param shape_calc_mode Specifies which input, sizes or scales, is used to calculate an output shape.
    shape_calculation_mode shape_calc_mode;
    /// @param antialias is a flag that specifies whether to perform anti-aliasing.
    int32_t antialias;
    /// @param cube_coeff specifies the parameter a for cubic interpolation. cube_coeff is used only when mode == cubic.
    float cube_coeff;
    /// @param coord_trans_mode specifies how to transform the coordinate in the resized tensor to the coordinate in the original tensor
    coordinate_transformation_mode coord_trans_mode;
    /// @param round_mode specifies round mode when mode == nearest and is used only when mode == nearest.
    nearest_mode round_mode;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
