// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/interpolate.hpp"

#include <numeric>

namespace ov {
namespace reference {

std::function<int64_t(float, bool)> get_func(Nearest_mode mode) {
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
    default:
        return [](float x_original, bool) {
            if (x_original == static_cast<int64_t>(x_original) + 0.5f) {
                return static_cast<int64_t>(std::floor(x_original));
            }
            return static_cast<int64_t>(std::round(x_original));
        };
    }
}

std::function<float(float, float, float, float)> get_func(Transform_mode mode) {
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
    default:
        return [](float x_resized, float x_scale, float, float) {
            return ((x_resized + 0.5f) / x_scale) - 0.5f;
        };
    }
}

float InterpolateEvalHelper::triangle_coeff(float dz) {
    return std::max(0.0f, 1.0f - std::fabs(dz));
}

std::array<float, 4> InterpolateEvalHelper::get_cubic_coeff(float s, float a) {
    std::array<float, 4> coeff;
    float abs_s = std::fabs(s);
    coeff[0] = ((a * (abs_s + 1) - 5 * a) * (abs_s + 1) + 8 * a) * (abs_s + 1) - 4 * a;
    coeff[1] = ((a + 2) * abs_s - (a + 3)) * abs_s * abs_s + 1;
    coeff[2] = ((a + 2) * (1 - abs_s) - (a + 3)) * (1 - abs_s) * (1 - abs_s) + 1;
    coeff[3] = ((a * (2 - abs_s) - 5 * a) * (2 - abs_s) + 8 * a) * (2 - abs_s) - 4 * a;
    return coeff;
}

Coordinate InterpolateEvalHelper::get_input_coords_for_nearest_mode(const Coordinate& output_coord) {
    std::size_t input_rank = m_input_data_shape.size();
    auto input_coord = output_coord;
    for (std::size_t i = 0; i < input_rank; ++i) {
        float length_original = static_cast<float>(m_input_data_shape[i]);
        float in_coord = m_get_original_coord(static_cast<float>(output_coord[i]),
                                              m_all_scales[i],
                                              static_cast<float>(m_out_shape[i]),
                                              length_original);
        int64_t nearest_pixel = m_get_nearest_pixel(in_coord, m_all_scales[i] < 1.0);
        input_coord[i] =
            std::max(static_cast<int64_t>(0), std::min(nearest_pixel, static_cast<int64_t>(length_original) - 1));
    }

    return input_coord;
}

InterpolateEvalHelper::InfoForGenericLinearONNXMode InterpolateEvalHelper::get_info_for_generic_linear_onnx() {
    InfoForGenericLinearONNXMode result;

    std::size_t input_rank = m_input_data_shape.size();

    Shape input_shape;
    Shape output_shape;

    switch (input_rank) {
    case 2:
        input_shape = Shape{1, 1, m_input_data_shape[0], m_input_data_shape[1]};
        output_shape = Shape{1, 1, m_out_shape[0], m_out_shape[1]};
        break;
    case 3:
        input_shape = Shape{1, 1, m_input_data_shape[0], m_input_data_shape[1], m_input_data_shape[2]};
        output_shape = Shape{1, 1, m_out_shape[0], m_out_shape[1], m_out_shape[2]};
        break;
    default:
        input_shape = m_input_data_shape;
        output_shape = m_out_shape;
    }

    int64_t batch_size = input_shape[0];
    int64_t num_channels = input_shape[1];

    std::size_t spatial_rank = input_shape.size() - 2;

    std::vector<int64_t> input_index_multipliers(spatial_rank);
    std::vector<int64_t> output_index_multipliers(spatial_rank);
    input_index_multipliers[spatial_rank - 1] = 1;
    output_index_multipliers[spatial_rank - 1] = 1;

    for (int64_t i = static_cast<int64_t>(spatial_rank) - 2; i >= 0; --i) {
        input_index_multipliers[i] = input_index_multipliers[i + 1] * static_cast<int64_t>(input_shape[i + 3]);
        output_index_multipliers[i] = output_index_multipliers[i + 1] * static_cast<int64_t>(output_shape[i + 3]);
    }

    int64_t input_data_ptr_increment = input_index_multipliers[0] * static_cast<int64_t>(input_shape[2]);
    int64_t output_data_ptr_increment = output_index_multipliers[0] * static_cast<int64_t>(output_shape[2]);

    std::vector<int64_t> input_spatial_shape(spatial_rank);
    std::vector<int64_t> output_spatial_shape(spatial_rank);

    for (size_t i = 0; i < spatial_rank; ++i) {
        input_spatial_shape[i] = static_cast<int64_t>(input_shape[i + 2]);
        output_spatial_shape[i] = static_cast<int64_t>(output_shape[i + 2]);
    }

    result.input_data_ptr_increment = input_data_ptr_increment;
    result.output_data_ptr_increment = output_data_ptr_increment;
    result.batch_size = batch_size;
    result.num_channels = num_channels;
    result.spatial_rank = static_cast<int64_t>(spatial_rank);
    result.input_index_multipliers = std::move(input_index_multipliers);
    result.output_index_multipliers = std::move(output_index_multipliers);
    result.input_spatial_shape = std::move(input_spatial_shape);
    result.output_spatial_shape = std::move(output_spatial_shape);

    return result;
}

float InterpolateEvalHelper::get_in_coord(float coord, int64_t axis_idx) {
    float scale = m_scales[axis_idx];
    int64_t axis = m_axes[axis_idx];
    float length_resized = static_cast<float>(m_out_shape[axis]);
    float length_original = static_cast<float>(m_input_data_shape[axis]);
    return m_get_original_coord(coord, scale, length_resized, length_original);
}

InterpolateEvalHelper::InfoForLinearMode InterpolateEvalHelper::get_info_for_linear_mode() {
    std::size_t num_of_axes = m_axes.size();
    bool is_downsample = false;
    for (const auto& scale : m_scales) {
        is_downsample = is_downsample || (scale < 1.0f);
    }

    bool antialias = is_downsample && m_antialias;

    std::vector<float> a(num_of_axes);
    std::vector<int64_t> r(num_of_axes);

    std::vector<std::size_t> vector_for_indices(num_of_axes);
    float prod_a = 1;
    for (std::size_t i = 0; i < num_of_axes; ++i) {
        a[i] = antialias ? m_scales[i] : 1.0f;
        prod_a *= a[i];
        r[i] = (m_scales[i] > 1.0) ? static_cast<int64_t>(2) : static_cast<int64_t>(std::ceil(2.0f / a[i]));
        vector_for_indices[i] = 2 * r[i] + 1;
    }
    Shape shape_for_indices{vector_for_indices};

    InfoForLinearMode result;

    result.antialias = antialias;
    result.a = std::move(a);
    result.r = std::move(r);
    result.prod_a = prod_a;
    result.shape_for_indices = std::move(shape_for_indices);

    return result;
}

InterpolateEvalHelper::ICoords InterpolateEvalHelper::get_icoords(const Coordinate& output_coord) {
    ICoords result;

    std::size_t input_rank = m_input_data_shape.size();
    std::size_t num_of_axes = m_axes.size();

    std::vector<float> icoords(input_rank);
    std::vector<int64_t> icoords_r(input_rank);
    for (std::size_t i = 0; i < input_rank; ++i) {
        icoords[i] = static_cast<float>(output_coord[i]);
        icoords_r[i] = output_coord[i];
    }

    for (std::size_t i = 0; i < num_of_axes; ++i) {
        int64_t axis = m_axes[i];
        float coordinate = static_cast<float>(output_coord[axis]);
        float in_coord = get_in_coord(coordinate, i);
        icoords[axis] = in_coord;
        icoords_r[axis] = static_cast<int64_t>(std::round(in_coord));
    }

    result.icoords = std::move(icoords);
    result.icoords_r = std::move(icoords_r);

    return result;
}

InterpolateEvalHelper::LinearModeInnerIterationResult InterpolateEvalHelper::inner_calculation(
    const Coordinate& output_coord,
    const ICoords& icoords_data,
    const InfoForLinearMode& info,
    const Coordinate& index) {
    std::size_t input_rank = m_input_data_shape.size();
    std::size_t num_of_axes = m_axes.size();

    LinearModeInnerIterationResult result;

    std::vector<size_t> inner_coords_vector(input_rank);
    for (std::size_t i = 0; i < input_rank; ++i) {
        inner_coords_vector[i] = output_coord[i];
    }

    for (std::size_t i = 0; i < num_of_axes; ++i) {
        int64_t axis = m_axes[i];
        inner_coords_vector[axis] = index[i] + icoords_data.icoords_r[axis] - info.r[i];
    }

    bool condition = true;
    for (int64_t axis : m_axes) {
        condition = condition && (inner_coords_vector[axis] < m_input_data_shape[axis]);
    }

    result.condition = condition;
    if (!condition) {
        return result;
    }

    std::vector<float> dz(num_of_axes);
    for (std::size_t i = 0; i < num_of_axes; ++i) {
        int64_t axis = m_axes[i];
        dz[i] = icoords_data.icoords[axis] - inner_coords_vector[axis];
    }

    float w = info.prod_a;
    for (std::size_t i = 0; i < num_of_axes; ++i) {
        w *= triangle_coeff(info.a[i] * dz[i]);
    }

    std::vector<std::size_t> unsigned_inner_coords_vector(input_rank);
    for (std::size_t i = 0; i < input_rank; ++i) {
        unsigned_inner_coords_vector[i] = inner_coords_vector[i];
    }

    Coordinate inner_coord{unsigned_inner_coords_vector};

    result.w = w;
    result.inner_coord = std::move(inner_coord);

    return result;
}

void pad_input_data(const uint8_t* data_ptr,
                    uint8_t* padded_data_ptr,
                    size_t type_size,
                    const ov::Shape& input_shape,
                    const ov::Shape& padded_input_shape,
                    const std::vector<size_t>& pads_begin) {
    const CoordinateTransformBasic input_transform{input_shape};

    for (const Coordinate& input_coord : input_transform) {
        auto padded_coord = input_coord;
        size_t i = 0;
        for (size_t pad : pads_begin) {
            padded_coord[i] += pad;
            ++i;
        }
        uint8_t* dst_ptr = padded_data_ptr + type_size * coordinate_index(padded_coord, padded_input_shape);
        const uint8_t* src_ptr = data_ptr + type_size * coordinate_index(input_coord, input_shape);
        memcpy(dst_ptr, src_ptr, type_size);
    }
}

PartialShape get_padded_input_shape(const PartialShape& input_shape, const op::v0::Interpolate::Attributes& attrs) {
    const auto input_rank = input_shape.rank().get_length();

    PartialShape padded_input_shape = input_shape;

    auto pads_begin = attrs.pads_begin;
    auto pads_end = attrs.pads_end;
    pads_begin.resize(input_rank);
    pads_end.resize(input_rank);
    for (int64_t i = 0; i < input_rank; ++i) {
        if (input_shape[i].is_static()) {
            auto new_length = pads_begin[i] + pads_end[i] + input_shape[i].get_length();
            padded_input_shape[i] = Dimension(new_length);
        }
    }

    return padded_input_shape;
}

std::vector<float> get_scales(const PartialShape& input_data_partial_shape,
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

op::v4::Interpolate::InterpolateAttrs transform_v0_to_v4(const PartialShape& input_partial_shape,
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
}  // namespace reference
}  // namespace ov
