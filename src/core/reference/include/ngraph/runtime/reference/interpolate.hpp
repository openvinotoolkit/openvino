// Copyright (C) 2018-2022 Intel Corporation
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

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/interpolate.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
using Nearest_mode = ngraph::op::v4::Interpolate::NearestMode;
using Transform_mode = ngraph::op::v4::Interpolate::CoordinateTransformMode;
using InterpolateMode = ngraph::op::v4::Interpolate::InterpolateMode;

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

    /// \brief Gets the function to calculate the nearest pixel.
    ///
    /// \param mode the calculation mode
    ///
    /// \return The function to calculate the nearest pixel.
    Func get_func(Nearest_mode mode) {
        switch (mode) {
        case Nearest_mode::ROUND_PREFER_CEIL:
            return [](float x_original, bool) {
                return static_cast<int64_t>(std::round(x_original));
            };
        case Nearest_mode::FLOOR:
            return [](float x_original, bool) {
                return static_cast<int64_t>(std::floor(x_original));
            };
        case Nearest_mode::CEIL:
            return [](float x_original, bool) {
                return static_cast<int64_t>(std::ceil(x_original));
            };
        case Nearest_mode::SIMPLE:
            return [](float x_original, bool is_downsample) {
                if (is_downsample) {
                    return static_cast<int64_t>(std::ceil(x_original));
                } else {
                    return static_cast<int64_t>(x_original);
                }
            };
        default:;
        }
        return [](float x_original, bool) {
            if (x_original == static_cast<int64_t>(x_original) + 0.5f) {
                return static_cast<int64_t>(std::floor(x_original));
            }
            return static_cast<int64_t>(std::round(x_original));
        };
    }
};

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

    /// \brief Gets the function to calculate the source coordinate.
    ///
    /// \param mode the calculation mode
    ///
    /// \return The function to calculate the source coordinate.
    Func get_func(Transform_mode mode) {
        switch (mode) {
        case Transform_mode::PYTORCH_HALF_PIXEL:
            return [](float x_resized, float x_scale, float length_resized, float) {
                return length_resized > 1 ? (x_resized + 0.5f) / x_scale - 0.5f : 0.0f;
            };
            break;
        case Transform_mode::ASYMMETRIC:
            return [](float x_resized, float x_scale, float, float) {
                return x_resized / x_scale;
            };
            break;
        case Transform_mode::TF_HALF_PIXEL_FOR_NN:
            return [](float x_resized, float x_scale, float, float) {
                return (x_resized + 0.5f) / x_scale;
            };
            break;
        case Transform_mode::ALIGN_CORNERS:
            return [](float x_resized, float, float length_resized, float length_original) {
                return length_resized == 1 ? 0 : x_resized * (length_original - 1) / (length_resized - 1);
            };
            break;
        default:;
        }
        return [](float x_resized, float x_scale, float, float) {
            return ((x_resized + 0.5f) / x_scale) - 0.5f;
        };
    }
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
        Shape shape_for_indeces;
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
};

template <typename T>
void InterpolateEval<T>::linear_func(const T* input_data, T* out) {
    auto info = helper.get_info_for_linear_mode();

    NGRAPH_SUPPRESS_DEPRECATED_START
    CoordinateTransform output_transform(m_out_shape);
    CoordinateTransform input_transform(m_input_data_shape);

    for (const Coordinate& output_coord : output_transform) {
        auto icoords_data = helper.get_icoords(output_coord);

        float summa = 0.0f;
        float wsum = 0.0f;

        CoordinateTransform indices{info.shape_for_indeces};
        for (const auto& index : indices) {
            auto inner_result = helper.inner_calculation(output_coord, icoords_data, info, index);
            if (!inner_result.condition) {
                continue;
            }

            wsum += inner_result.w;
            summa += inner_result.w * static_cast<float>(input_data[input_transform.index(inner_result.inner_coord)]);
        }

        if (wsum == 0.0f) {
            out[output_transform.index(output_coord)] = T{};
        } else {
            if (std::is_integral<T>()) {
                // Round value for integral return types
                out[output_transform.index(output_coord)] = static_cast<T>(std::round(summa / wsum));
            } else {
                out[output_transform.index(output_coord)] = static_cast<T>(summa / wsum);
            }
        }
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
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
        throw ngraph_error("Axes are not correct!");

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

    NGRAPH_SUPPRESS_DEPRECATED_START
    CoordinateTransform output_transform(m_out_shape);
    CoordinateTransform input_transform(m_input_data_shape);
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
        CoordinateTransform indices{indices_shape};

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

            summa += coeffs_prod * static_cast<float>(input_data[input_transform.index(coords_for_sum)]);
        }

        out[output_transform.index(output_coord)] = static_cast<T>(summa);
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}

template <typename T>
void InterpolateEval<T>::nearest_func(const T* input_data, T* out) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    CoordinateTransform output_transform(m_out_shape);
    CoordinateTransform input_transform(m_input_data_shape);

    for (const Coordinate& output_coord : output_transform) {
        auto input_coord = helper.get_input_coords_for_nearest_mode(output_coord);
        out[output_transform.index(output_coord)] = input_data[input_transform.index(input_coord)];
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}

static void pad_input_data(const uint8_t* data_ptr,
                           uint8_t* padded_data_ptr,
                           size_t type_size,
                           const ov::Shape& input_shape,
                           const ov::Shape& padded_input_shape,
                           const std::vector<size_t>& pads_begin) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    CoordinateTransform input_transform(input_shape);
    CoordinateTransform padded_transform(padded_input_shape);

    for (const Coordinate& input_coord : input_transform) {
        auto padded_coord = input_coord;
        size_t i = 0;
        for (size_t pad : pads_begin) {
            padded_coord[i] += pad;
            ++i;
        }
        uint8_t* dst_ptr = padded_data_ptr + type_size * padded_transform.index(padded_coord);
        const uint8_t* src_ptr = data_ptr + type_size * input_transform.index(input_coord);
        memcpy(dst_ptr, src_ptr, type_size);
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}

static PartialShape get_padded_input_shape(const PartialShape& input_shape,
                                           const op::v0::Interpolate::Attributes& attrs) {
    const auto input_rank = input_shape.rank().get_length();

    PartialShape padded_input_shape = input_shape;

    for (int64_t i = 0; i < input_rank; ++i) {
        if (input_shape[i].is_static()) {
            auto new_length = attrs.pads_begin[i] + attrs.pads_end[i] + input_shape[i].get_length();
            padded_input_shape[i] = Dimension(new_length);
        }
    }

    return padded_input_shape;
}

static std::vector<float> get_scales(const PartialShape& input_data_partial_shape,
                                     const Shape& out_shape,
                                     const op::v0::Interpolate::Attributes& attrs) {
    std::vector<float> scales(attrs.axes.size(), 1.0f);
    auto input_shape = input_data_partial_shape.to_shape();
    size_t i = 0;
    for (size_t axis : attrs.axes) {
        scales[i] = static_cast<float>(out_shape.at(axis)) / input_shape.at(axis);
        i++;
    }

    return scales;
}

static op::v4::Interpolate::InterpolateAttrs transform_v0_to_v4(const PartialShape& input_partial_shape,
                                                                const op::v0::Interpolate::Attributes& attrs_v0) {
    auto input_shape_rank = input_partial_shape.rank().get_length();

    op::v4::Interpolate::InterpolateAttrs attrs_v4;
    if (attrs_v0.mode == "nearest") {
        attrs_v4.mode = InterpolateMode::NEAREST;
    } else if (attrs_v0.mode == "linear") {
        if (input_shape_rank < 5) {
            attrs_v4.mode = InterpolateMode::LINEAR_ONNX;
        } else if (input_shape_rank == 5) {
            attrs_v4.mode = InterpolateMode::LINEAR;
        } else {
            OPENVINO_ASSERT(false, "Failed to process ", attrs_v0.mode);
        }
    } else if (attrs_v0.mode == "cubic") {
        attrs_v4.mode = InterpolateMode::CUBIC;
    } else if (attrs_v0.mode == "linear_onnx") {
        attrs_v4.mode = InterpolateMode::LINEAR_ONNX;
    } else {
        OPENVINO_ASSERT(false, "Failed to process ", attrs_v0.mode);
    }

    attrs_v4.shape_calculation_mode = op::v4::Interpolate::ShapeCalcMode::SIZES;
    attrs_v4.nearest_mode = Nearest_mode::SIMPLE;
    attrs_v4.pads_begin = attrs_v0.pads_begin;
    attrs_v4.pads_end = attrs_v0.pads_end;
    attrs_v4.antialias = attrs_v0.antialias;
    attrs_v4.coordinate_transformation_mode = Transform_mode::ASYMMETRIC;
    attrs_v4.cube_coeff = -0.75f;

    if (attrs_v0.align_corners) {
        attrs_v4.coordinate_transformation_mode = Transform_mode::ALIGN_CORNERS;
    } else if ((attrs_v4.mode == InterpolateMode::LINEAR_ONNX || attrs_v4.mode == InterpolateMode::LINEAR) &&
               std::all_of(attrs_v4.pads_begin.begin(),
                           attrs_v4.pads_begin.end(),
                           [](size_t i) {
                               return i == 0;
                           }) &&
               std::all_of(attrs_v4.pads_end.begin(),
                           attrs_v4.pads_end.end(),
                           [](size_t i) {
                               return i == 0;
                           }) &&
               !(input_shape_rank - 2 == 2 && attrs_v0.axes == AxisSet{2, 3})) {
        attrs_v4.coordinate_transformation_mode = Transform_mode::HALF_PIXEL;
    }

    return attrs_v4;
}

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
}  // namespace runtime
}  // namespace ngraph
