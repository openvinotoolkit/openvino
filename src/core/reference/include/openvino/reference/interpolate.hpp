// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>

#include "interpolate_pil.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"
#include "transpose.hpp"

namespace ov {
namespace reference {
using Nearest_mode = op::v4::Interpolate::NearestMode;
using Transform_mode = op::v4::Interpolate::CoordinateTransformMode;
using InterpolateMode = op::v4::Interpolate::InterpolateMode;

/// \brief Gets the function to calculate the nearest pixel.
///
/// \param mode the calculation mode
///
/// \return The function to calculate the nearest pixel.
std::function<int64_t(float, bool)> get_func(Nearest_mode mode);
/// \brief Calculation of nearest pixel.
class GetNearestPixel final {
public:
    /// \brief Constructs calculation of a nearest pixel in the default mode.
    GetNearestPixel() : GetNearestPixel(Nearest_mode::ROUND_PREFER_FLOOR) {}

    /// \brief Constructs calculation of nearest pixel for the specified mode.
    ///
    /// \param mode the mode of the calculation of the nearest pixel
    GetNearestPixel(Nearest_mode mode) : m_func{get_func(mode)} {}

    ~GetNearestPixel() = default;

    /// \brief Performing the nearest pixel calculation.
    ///
    /// \param original original coordinate
    /// \param is_downsample true if it has downsample and false otherwise
    ///
    /// \return the nearest pixel
    int64_t operator()(float original, bool is_downsample) const {
        return m_func(original, is_downsample);
    }

private:
    using Func = std::function<int64_t(float, bool)>;

    Func m_func;
};

/// \brief Gets the function to calculate the source coordinate.
///
/// \param mode the calculation mode
///
/// \return The function to calculate the source coordinate.
std::function<float(float, float, float, float)> get_func(Transform_mode mode);

/// \brief Calculation of the source coordinate using the resized coordinate
class GetOriginalCoordinate final {
public:
    /// \brief Constructs calculation of a nearest pixel in the default mode.
    GetOriginalCoordinate() : GetOriginalCoordinate(Transform_mode::HALF_PIXEL) {}

    /// \brief Constructs calculation of the source coordinate.
    ///
    /// \param mode the mode of the calculation of the source coordinate.
    GetOriginalCoordinate(Transform_mode mode) : m_func{get_func(mode)} {}

    ~GetOriginalCoordinate() = default;

    /// \brief Performing the source coordinate calculation.
    ///
    /// \param x_resized resized coordinate
    /// \param x_scale scale for the considered axis
    /// \param length_resized length of the resized axis
    /// \param length_original original length of the axis
    ///
    /// \return the source coordinate
    float operator()(float x_resized, float x_scale, float length_resized, float length_original) const {
        if (x_scale == 1.0f || (length_resized == length_original)) {
            return x_resized;
        }
        return m_func(x_resized, x_scale, length_resized, length_original);
    }

private:
    using Func = std::function<float(float, float, float, float)>;

    Func m_func;
};

/// \brief Helper class to implent non-template parts of the interpolation calculation.
class InterpolateEvalHelper final {
public:
    InterpolateEvalHelper() = default;

    InterpolateEvalHelper(const op::v4::Interpolate::InterpolateAttrs& attrs,
                          const Shape& input_data_shape,
                          const std::vector<int64_t>& axes,
                          const Shape& out_shape,
                          const std::vector<float>& scales)
        : m_get_nearest_pixel{attrs.nearest_mode},
          m_get_original_coord{attrs.coordinate_transformation_mode},
          m_antialias{attrs.antialias},
          m_input_data_shape{input_data_shape},
          m_axes{axes},
          m_out_shape{out_shape},
          m_scales{scales} {
        size_t input_rank = input_data_shape.size();
        m_all_scales = std::vector<float>(input_rank, 1.0f);
        size_t num_of_axes = axes.size();

        for (size_t i = 0; i < num_of_axes; ++i) {
            m_all_scales[axes[i]] = scales[i];
        }
    }

    ~InterpolateEvalHelper() = default;

    float triangle_coeff(float dz);
    std::array<float, 4> get_cubic_coeff(float s, float a);
    float get_in_coord(float coord, int64_t axis_idx);

    Coordinate get_input_coords_for_nearest_mode(const Coordinate& output_coord);

    struct InfoForGenericLinearONNXMode {
        int64_t input_data_ptr_increment;
        int64_t output_data_ptr_increment;
        int64_t batch_size;
        int64_t num_channels;
        int64_t spatial_rank;
        std::vector<int64_t> input_index_multipliers;
        std::vector<int64_t> output_index_multipliers;
        std::vector<int64_t> input_spatial_shape;
        std::vector<int64_t> output_spatial_shape;
    };

    InfoForGenericLinearONNXMode get_info_for_generic_linear_onnx();

    struct InfoForLinearMode {
        bool antialias;
        float prod_a;
        std::vector<float> a;
        std::vector<int64_t> r;
        Shape shape_for_indices;
    };

    InfoForLinearMode get_info_for_linear_mode();

    struct ICoords {
        std::vector<float> icoords;
        std::vector<int64_t> icoords_r;
    };

    ICoords get_icoords(const Coordinate& output_coord);

    struct LinearModeInnerIterationResult {
        bool condition;
        float w = 0;
        Coordinate inner_coord;
    };

    LinearModeInnerIterationResult inner_calculation(const Coordinate& output_coord,
                                                     const ICoords& icoords_data,
                                                     const InfoForLinearMode& info,
                                                     const Coordinate& index);

private:
    GetNearestPixel m_get_nearest_pixel;
    GetOriginalCoordinate m_get_original_coord;
    bool m_antialias{false};

    Shape m_input_data_shape;
    std::vector<int64_t> m_axes;
    Shape m_out_shape;

    std::vector<float> m_scales;
    std::vector<float> m_all_scales;
};

/// \brief Class to perform interpolation calculation.
template <typename T>
class InterpolateEval final {
public:
    InterpolateEval() = default;

    /// \brief Constructs interpolation calculation using Interpolate attributes.
    ///
    /// \param attrs Interpolate-4 attributes.
    InterpolateEval(const op::v4::Interpolate::InterpolateAttrs& attrs)
        : m_attrs{attrs},
          m_interp_mode{attrs.mode},
          m_cube_coeff{attrs.cube_coeff} {}

    ~InterpolateEval() = default;

    /// \brief Performing interpolation calculation.
    ///
    /// \param input_data pointer to input data
    /// \param input_data_shape shape of the input data
    /// \param scales scale factors for each interpolated axis
    /// \param axes axes to interpolate
    /// \param out pointer to memory block for output data
    /// \param out_shape shape of output data
    void operator()(const T* input_data,
                    const Shape& input_data_shape,
                    const std::vector<float>& scales,
                    const std::vector<int64_t>& axes,
                    T* out,
                    const Shape& out_shape) {
        assert(axes.size() == scales.size());

        m_input_data_shape = input_data_shape;
        m_axes = axes;
        m_out_shape = out_shape;

        size_t output_data_size = shape_size(out_shape);
        std::fill(out, out + output_data_size, T{});

        m_scales = scales;

        helper = InterpolateEvalHelper{m_attrs, input_data_shape, axes, out_shape, scales};

        switch (m_interp_mode) {
        case InterpolateMode::NEAREST:
            nearest_func(input_data, out);
            break;
        case InterpolateMode::LINEAR:
            linear_func(input_data, out);
            break;
        case InterpolateMode::LINEAR_ONNX:
            linear_onnx_func(input_data, out);
            break;
        case InterpolateMode::CUBIC:
            cubic_func(input_data, out);
            break;
        case InterpolateMode::BILINEAR_PILLOW:
            bilinear_pil_func(input_data, out);
            break;
        case InterpolateMode::BICUBIC_PILLOW:
            bicubic_pil_func(input_data, out);
            break;
        default:
            OPENVINO_THROW("Unsupported interpolation mode");
            break;
        }
    }

private:
    op::v4::Interpolate::InterpolateAttrs m_attrs;

    InterpolateMode m_interp_mode;
    double m_cube_coeff;

    Shape m_input_data_shape;
    std::vector<int64_t> m_axes;
    Shape m_out_shape;

    std::vector<float> m_scales;

    InterpolateEvalHelper helper;

    /// \brief Calculates linear interpolation.
    ///
    /// \param input_data pointer to input data
    /// \param out pointer to memory block for output data
    void linear_func(const T* input_data, T* out);

    /// \brief Calculates interpolation as in ONNX 'linear' mode (generic case)
    ///
    /// \param input_data pointer to input data
    /// \param out pointer to memory block for output data
    void linear_onnx_func(const T* input_data, T* out);

    /// \brief Calculates cubic interpolation.
    ///
    /// \param input_data pointer to input data
    /// \param out pointer to memory block for output data
    void cubic_func(const T* input_data, T* out);

    /// \brief Calculates 'nearest' mode of interpolation.
    ///
    /// \param input_data pointer to input data
    /// \param out pointer to memory block for output data
    void nearest_func(const T* input_data, T* out);

    void bilinear_pil_func(const T* input_data, T* out);
    void bicubic_pil_func(const T* input_data, T* out);
    void multidim_pil_func(const T* input_data, T* out, const interpolate_pil::filter& filterp);
};

template <typename T>
void InterpolateEval<T>::linear_func(const T* input_data, T* out) {
    auto info = helper.get_info_for_linear_mode();

    const CoordinateTransformBasic output_transform{m_out_shape};

    for (const Coordinate& output_coord : output_transform) {
        auto icoords_data = helper.get_icoords(output_coord);

        float summa = 0.0f;
        float wsum = 0.0f;

        const CoordinateTransformBasic indices{info.shape_for_indices};
        for (const auto& index : indices) {
            auto inner_result = helper.inner_calculation(output_coord, icoords_data, info, index);
            if (!inner_result.condition) {
                continue;
            }

            const auto input_index = coordinate_index(inner_result.inner_coord, m_input_data_shape);
            wsum += inner_result.w;
            summa += inner_result.w * static_cast<float>(input_data[input_index]);
        }

        const auto out_index = coordinate_index(output_coord, m_out_shape);
        if (wsum == 0.0f) {
            out[out_index] = T{};
        } else {
            if (std::is_integral<T>()) {
                // Round value for integral return types
                out[out_index] = static_cast<T>(std::round(summa / wsum));
            } else {
                out[out_index] = static_cast<T>(summa / wsum);
            }
        }
    }
}

template <typename T>
void InterpolateEval<T>::linear_onnx_func(const T* input_data, T* out) {
    const size_t input_rank = m_input_data_shape.size();

    assert(input_rank > 1);

    const size_t num_of_axes = m_axes.size();

    bool correct_axes =
        ((input_rank == 2) && (num_of_axes == 2) && (m_axes[0] == 0) && (m_axes[1] == 1)) ||
        ((input_rank == 3) && (num_of_axes == 3) && (m_axes[0] == 0) && (m_axes[1] == 1) && (m_axes[2] == 2));

    if (input_rank >= 4) {
        std::vector<int64_t> all_axes;
        std::vector<int64_t> axes_without_batch_and_channels;
        all_axes.push_back(0);
        all_axes.push_back(1);
        for (int64_t i = 2; i < static_cast<int64_t>(input_rank); ++i) {
            all_axes.push_back(i);
            axes_without_batch_and_channels.push_back(i);
        }

        correct_axes = ((num_of_axes == input_rank) && (m_axes == all_axes)) ||
                       ((num_of_axes == input_rank - 2) && (m_axes == axes_without_batch_and_channels));
    }

    if (!correct_axes)
        OPENVINO_THROW("Axes are not correct!");

    const auto info = helper.get_info_for_generic_linear_onnx();

    const int64_t batch_size = info.batch_size;
    const int64_t num_channels = info.num_channels;

    const auto& input_index_multipliers = info.input_index_multipliers;
    const auto& output_index_multipliers = info.output_index_multipliers;

    const int64_t input_data_ptr_increment = info.input_data_ptr_increment;
    const int64_t output_data_ptr_increment = info.output_data_ptr_increment;

    const auto& input_spatial_shape = info.input_spatial_shape;

    // This mode supports only interpolation with respect to spatial dimensions,
    // not with respect to batch or channels. That is, we can have only two cases:
    //     num_of_axes == input_rank
    // or
    //     num_of_axes == input_rank - 2.
    // Hence, if num_of_axes != input_rank, then interpolated axes indices are
    //    [0, 1, ..., num_of_axes - 1]
    // Otherwise, if num_of_axes == input_rank, interpolated axes indices are
    //    [2, 3, ..., num_of_axes - 1]
    const int64_t axis_idx_offset = (input_rank == num_of_axes) ? 2 : 0;

    const int64_t spatial_rank = info.spatial_rank;
    const int64_t points_in_neighbor = 1LL << spatial_rank;

    const T* xdata = input_data;
    T* ydata = out;
    for (int64_t n = 0; n < batch_size; ++n) {
        for (int64_t c = 0; c < num_channels; ++c) {
            for (int64_t idx = 0; idx < output_data_ptr_increment; ++idx) {
                // 1. Get the current spatial coords vector.
                std::vector<int64_t> output_coords(spatial_rank);
                int64_t curr = idx;
                for (int64_t j = 0; j < spatial_rank - 1; ++j) {
                    output_coords[j] = curr / output_index_multipliers[j];
                    curr %= output_index_multipliers[j];
                }
                output_coords[spatial_rank - 1] = curr;

                // 2. Some preliminaries.
                std::vector<int64_t> in1(spatial_rank);
                std::vector<int64_t> in2(spatial_rank);
                std::vector<float> d1(spatial_rank);
                std::vector<float> d2(spatial_rank);

                for (int64_t i = 0; i < spatial_rank; ++i) {
                    float out_coord = static_cast<float>(output_coords[i]);

                    float in_coord = helper.get_in_coord(out_coord, i + axis_idx_offset);
                    in_coord = std::max(0.0f, std::min(in_coord, static_cast<float>(input_spatial_shape[i] - 1)));

                    const int64_t in_coord1 = std::min(static_cast<int64_t>(in_coord), input_spatial_shape[i] - 1);
                    const int64_t in_coord2 = std::min(in_coord1 + 1, input_spatial_shape[i] - 1);

                    in1[i] = in_coord1;
                    in2[i] = in_coord2;
                    d1[i] = std::fabs(in_coord - in_coord1);
                    d2[i] = std::fabs(in_coord - in_coord2);

                    if (in_coord1 == in_coord2) {
                        d1[i] = 0.5f;
                        d2[i] = 0.5f;
                    }
                }

                // 3. Get values in all points of a neighborhood.
                std::vector<T> values_of_input_points(points_in_neighbor);
                for (int64_t i = 0; i < points_in_neighbor; ++i) {
                    int64_t offset = 0;
                    for (int64_t j = 0; j < spatial_rank; ++j) {
                        if (i & (static_cast<int64_t>(1) << (spatial_rank - 1 - j))) {
                            offset += in1[j] * input_index_multipliers[j];
                        } else {
                            offset += in2[j] * input_index_multipliers[j];
                        }
                    }
                    values_of_input_points[i] = xdata[offset];
                }

                // 4. Interpolation.
                float sum = 0.0f;
                for (int64_t i = 0; i < points_in_neighbor; ++i) {
                    float coeff = 1.0f;
                    for (int64_t j = 0; j < spatial_rank; ++j) {
                        coeff *= (i & (static_cast<int64_t>(1) << (spatial_rank - 1 - j))) ? d1[j] : d2[j];
                    }
                    sum += coeff * static_cast<float>(values_of_input_points[points_in_neighbor - 1 - i]);
                }

                // 6. Store result.
                ydata[idx] = static_cast<T>(sum);
            }

            xdata += input_data_ptr_increment;
            ydata += output_data_ptr_increment;
        }
    }
}

template <typename T>
void InterpolateEval<T>::cubic_func(const T* input_data, T* out) {
    size_t input_rank = m_input_data_shape.size();
    size_t num_of_axes = m_axes.size();

    const CoordinateTransformBasic output_transform{m_out_shape};
    Shape indices_shape{std::vector<size_t>(num_of_axes, 4)};

    for (const Coordinate& output_coord : output_transform) {
        std::map<size_t, std::array<float, 4>> cubic_coeffs;
        std::vector<int64_t> base_coords(input_rank, 0);
        for (size_t i = 0; i < num_of_axes; ++i) {
            int64_t axis = m_axes[i];
            float coordinate = static_cast<float>(output_coord[axis]);
            float in_coord = helper.get_in_coord(coordinate, i);
            int64_t in_coord_int = static_cast<int64_t>(std::floor(in_coord));
            base_coords[axis] = in_coord_int;
            auto s = static_cast<float>(in_coord - in_coord_int);
            cubic_coeffs[axis] = helper.get_cubic_coeff(s, static_cast<float>(m_cube_coeff));
        }

        float summa = 0.0f;
        const CoordinateTransformBasic indices{indices_shape};

        for (const Coordinate& idx : indices) {
            auto coords_for_sum = output_coord;
            float coeffs_prod = 1.0;
            for (size_t i = 0; i < num_of_axes; ++i) {
                int64_t axis = m_axes[i];
                int64_t coord_to_clip =
                    static_cast<int64_t>(base_coords[axis]) + static_cast<int64_t>(idx[i]) - static_cast<int64_t>(1);
                int64_t clipped_coord =
                    std::max(static_cast<int64_t>(0),
                             std::min(coord_to_clip, static_cast<int64_t>(m_input_data_shape[axis]) - 1));
                coords_for_sum[axis] = clipped_coord;
                coeffs_prod *= cubic_coeffs[axis][idx[i]];
            }

            const auto input_index = coordinate_index(coords_for_sum, m_input_data_shape);
            summa += coeffs_prod * static_cast<float>(input_data[input_index]);
        }

        out[coordinate_index(output_coord, m_out_shape)] = static_cast<T>(summa);
    }
}

template <typename T>
void InterpolateEval<T>::bilinear_pil_func(const T* input_data, T* out) {
    struct interpolate_pil::filter bilinear = {interpolate_pil::bilinear_filter, 1.0, m_cube_coeff};
    multidim_pil_func(input_data, out, bilinear);
}

template <typename T>
void InterpolateEval<T>::bicubic_pil_func(const T* input_data, T* out) {
    struct interpolate_pil::filter bicubic = {interpolate_pil::bicubic_filter, 2.0, m_cube_coeff};
    multidim_pil_func(input_data, out, bicubic);
}

template <typename T>
void InterpolateEval<T>::multidim_pil_func(const T* input_data, T* out, const interpolate_pil::filter& filterp) {
    OPENVINO_ASSERT(m_axes.size() == 2, "For Pillow based modes exactly two (HW) axes need to be provided.");

    auto h_dim_idx = m_axes[0];
    auto w_dim_idx = m_axes[1];
    auto h_dim_in = m_input_data_shape[h_dim_idx];
    auto w_dim_in = m_input_data_shape[w_dim_idx];
    auto h_dim_out = m_out_shape[h_dim_idx];
    auto w_dim_out = m_out_shape[w_dim_idx];
    auto in_matrix_elem_size = h_dim_in * w_dim_in;
    auto out_matrix_elem_size = h_dim_out * w_dim_out;

    auto box = std::vector<float>{0.f, 0.f, static_cast<float>(w_dim_in), static_cast<float>(h_dim_in)};

    if (shape_size(m_input_data_shape) == in_matrix_elem_size) {
        // Input data is 2D or ND with other dimensions equal 1
        interpolate_pil::imaging_resample_inner(input_data,
                                                w_dim_in,
                                                h_dim_in,
                                                w_dim_out,
                                                h_dim_out,
                                                filterp,
                                                box.data(),
                                                out);
    } else {
        // Flatten other dimensions and interpolate over 2D matrices
        std::vector<int64_t> in_transp_axes_order;
        for (size_t i = 0; i < m_input_data_shape.size(); ++i) {
            if (std::find(m_axes.begin(), m_axes.end(), i) == m_axes.end()) {
                in_transp_axes_order.push_back(i);
            }
        }
        in_transp_axes_order.insert(in_transp_axes_order.end(), m_axes.begin(), m_axes.end());

        Shape transp_input_shape;
        Shape transp_output_shape;
        for (auto&& axis : in_transp_axes_order) {
            transp_input_shape.push_back(m_input_data_shape[axis]);
            transp_output_shape.push_back(m_out_shape[axis]);
        }
        size_t flat_batch_size =
            transp_input_shape.size() > 2
                ? shape_size(transp_input_shape.begin(), transp_input_shape.begin() + transp_input_shape.size() - 2)
                : 1;

        // Transpose HW dimensions to the end of the tensor shape
        std::vector<T> transposed_in(input_data, input_data + shape_size(m_input_data_shape));
        transpose(reinterpret_cast<const char*>(input_data),
                  reinterpret_cast<char*>(transposed_in.data()),
                  m_input_data_shape,
                  sizeof(T),
                  in_transp_axes_order,
                  transp_input_shape);

        std::vector<T> transposed_out(shape_size(m_out_shape));
        T* in_matrix_ptr = transposed_in.data();
        T* out_matrix_ptr = transposed_out.data();

        // Resample each 2D matrix
        for (size_t i = 0; i < flat_batch_size; ++i) {
            interpolate_pil::imaging_resample_inner(in_matrix_ptr,
                                                    w_dim_in,
                                                    h_dim_in,
                                                    w_dim_out,
                                                    h_dim_out,
                                                    filterp,
                                                    box.data(),
                                                    out_matrix_ptr);
            in_matrix_ptr += in_matrix_elem_size;
            out_matrix_ptr += out_matrix_elem_size;
        }

        std::vector<int64_t> out_transp_axes_order(m_out_shape.size() - 2);
        std::iota(out_transp_axes_order.begin(), out_transp_axes_order.end(), 0);
        out_transp_axes_order.insert(out_transp_axes_order.begin() + h_dim_idx, transp_input_shape.size() - 2);
        out_transp_axes_order.insert(out_transp_axes_order.begin() + w_dim_idx, transp_input_shape.size() - 1);

        // Transpose back to the original data dimensions order
        transpose(reinterpret_cast<const char*>(transposed_out.data()),
                  reinterpret_cast<char*>(out),
                  transp_output_shape,
                  sizeof(T),
                  out_transp_axes_order,
                  m_out_shape);
    }
}

template <typename T>
void InterpolateEval<T>::nearest_func(const T* input_data, T* out) {
    const CoordinateTransformBasic output_transform{m_out_shape};

    for (const Coordinate& output_coord : output_transform) {
        auto input_coord = helper.get_input_coords_for_nearest_mode(output_coord);
        const auto input_index = coordinate_index(input_coord, m_input_data_shape);
        const auto out_index = coordinate_index(output_coord, m_out_shape);
        out[out_index] = input_data[input_index];
    }
}

void pad_input_data(const uint8_t* data_ptr,
                    uint8_t* padded_data_ptr,
                    size_t type_size,
                    const ov::Shape& input_shape,
                    const ov::Shape& padded_input_shape,
                    const std::vector<size_t>& pads_begin);

PartialShape get_padded_input_shape(const PartialShape& input_shape, const op::v0::Interpolate::Attributes& attrs);

std::vector<float> get_scales(const PartialShape& input_data_partial_shape,
                              const Shape& out_shape,
                              const op::v0::Interpolate::Attributes& attrs);
op::v4::Interpolate::InterpolateAttrs transform_v0_to_v4(const PartialShape& input_partial_shape,
                                                         const op::v0::Interpolate::Attributes& attrs_v0);

template <typename T>
void interpolate(const T* input_data,
                 const Shape& input_data_shape,
                 const std::vector<float>& scales,
                 const std::vector<int64_t>& axes,
                 T* out,
                 const Shape& out_shape,
                 const op::v4::Interpolate::InterpolateAttrs& attrs) {
    InterpolateEval<T> evaluator{attrs};
    evaluator(input_data, input_data_shape, scales, axes, out, out_shape);
}

template <typename T>
void interpolate(T* input_data,
                 const PartialShape& input_data_shape,
                 T* out,
                 const Shape& out_shape,
                 const op::v0::Interpolate::Attributes& attrs) {
    InterpolateEval<T> evaluator{transform_v0_to_v4(input_data_shape, attrs)};

    Shape padded_input_shape = get_padded_input_shape(input_data_shape, attrs).to_shape();
    std::vector<float> scales = get_scales(padded_input_shape, out_shape, attrs);
    std::vector<int64_t> axes{attrs.axes.begin(), attrs.axes.end()};

    size_t bytes_in_padded_input = shape_size(padded_input_shape) * sizeof(T);
    std::vector<uint8_t> padded_input_data(bytes_in_padded_input, 0);
    uint8_t* padded_data_ptr = padded_input_data.data();
    pad_input_data(reinterpret_cast<uint8_t*>(input_data),
                   padded_data_ptr,
                   sizeof(T),
                   input_data_shape.to_shape(),
                   padded_input_shape,
                   attrs.pads_begin);

    evaluator(reinterpret_cast<T*>(padded_data_ptr), padded_input_shape, scales, axes, out, out_shape);
}
}  // namespace reference
}  // namespace ov
