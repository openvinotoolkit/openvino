//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/interpolate.hpp"
#include "ngraph/runtime/reference/interpolate.hpp"

using namespace ngraph::runtime::reference;

using Coordinate = ngraph::Coordinate;

float InterpolateEvalHelper::triangle_coeff(float dz)
{
    return std::max(0.0f, 1.0f - std::fabs(dz));
}

std::array<float, 4> InterpolateEvalHelper::get_cubic_coeff(float s, float a)
{
    std::array<float, 4> coeff;
    float abs_s = std::fabs(s);
    coeff[0] = ((a * (abs_s + 1) - 5 * a) * (abs_s + 1) + 8 * a) * (abs_s + 1) - 4 * a;
    coeff[1] = ((a + 2) * abs_s - (a + 3)) * abs_s * abs_s + 1;
    coeff[2] = ((a + 2) * (1 - abs_s) - (a + 3)) * (1 - abs_s) * (1 - abs_s) + 1;
    coeff[3] = ((a * (2 - abs_s) - 5 * a) * (2 - abs_s) + 8 * a) * (2 - abs_s) - 4 * a;
    return coeff;
}

Coordinate InterpolateEvalHelper::get_input_coords_for_nearest_mode(const Coordinate& output_coord)
{
    std::size_t input_rank = m_input_data_shape.size();
    auto input_coord = output_coord;
    for (std::size_t i = 0; i < input_rank; ++i)
    {
        float length_original = static_cast<float>(m_input_data_shape[i]);
        float in_coord = m_get_original_coord(static_cast<float>(output_coord[i]),
                                              m_all_scales[i],
                                              static_cast<float>(m_out_shape[i]),
                                              length_original);
        int64_t nearest_pixel = m_get_nearest_pixel(in_coord, m_all_scales[i] < 1.0);
        input_coord[i] =
            std::max(static_cast<int64_t>(0),
                     std::min(nearest_pixel, static_cast<int64_t>(length_original) - 1));
    }

    return input_coord;
}

InterpolateEvalHelper::InfoForLinearONNXMode InterpolateEvalHelper::get_info_for_linear_onnx_mode()
{
    InfoForLinearONNXMode result;

    std::size_t input_rank = m_input_data_shape.size();
    std::size_t num_of_axes = m_axes.size();

    Shape input_shape = Shape{1, 1, m_input_data_shape[0], m_input_data_shape[1]};
    Shape output_shape = Shape{1, 1, m_out_shape[0], m_out_shape[1]};

    if (input_rank == 4)
    {
        input_shape = m_input_data_shape;
        output_shape = m_out_shape;
    }

    int64_t batch_size = input_shape[0];
    int64_t num_channels = input_shape[1];
    int64_t input_height = input_shape[2];
    int64_t input_width = input_shape[3];
    int64_t output_height = output_shape[2];
    int64_t output_width = output_shape[3];
    float height_scale = m_scales[0];
    float width_scale = m_scales[1];

    if (num_of_axes == 4)
    {
        height_scale = m_scales[2];
        width_scale = m_scales[3];
    }

    std::vector<float> y_original(output_height);
    std::vector<float> x_original(output_width);

    std::vector<int64_t> input_width_mul_y1(output_height);
    std::vector<int64_t> input_width_mul_y2(output_height);
    std::vector<int64_t> in_x1(output_width);
    std::vector<int64_t> in_x2(output_width);

    std::vector<float> dy1(output_height);
    std::vector<float> dy2(output_height);
    std::vector<float> dx1(output_width);
    std::vector<float> dx2(output_width);

    for (int64_t y = 0; y < output_height; ++y)
    {
        float in_y = m_get_original_coord(static_cast<float>(y),
                                          height_scale,
                                          static_cast<float>(output_height),
                                          static_cast<float>(input_height));
        y_original[y] = in_y;
        in_y = std::max(0.0f, std::min(in_y, static_cast<float>(input_height - 1)));

        const int64_t in_y1 = std::min(static_cast<int64_t>(in_y), input_height - 1);
        const int64_t in_y2 = std::min(in_y1 + 1, input_height - 1);
        dy1[y] = std::fabs(in_y - in_y1);
        dy2[y] = std::fabs(in_y - in_y2);

        if (in_y1 == in_y2)
        {
            dy1[y] = 0.5f;
            dy2[y] = 0.5f;
        }

        input_width_mul_y1[y] = input_width * in_y1;
        input_width_mul_y2[y] = input_width * in_y2;
    }

    for (int64_t x = 0; x < output_width; ++x)
    {
        float in_x = m_get_original_coord(static_cast<float>(x),
                                          width_scale,
                                          static_cast<float>(output_width),
                                          static_cast<float>(input_width));
        x_original[x] = in_x;
        in_x = std::max(0.0f, std::min(in_x, static_cast<float>(input_width - 1)));

        in_x1[x] = std::min(static_cast<int64_t>(in_x), input_width - 1);
        in_x2[x] = std::min(in_x1[x] + 1, input_width - 1);

        dx1[x] = std::abs(in_x - in_x1[x]);
        dx2[x] = std::abs(in_x - in_x2[x]);
        if (in_x1[x] == in_x2[x])
        {
            dx1[x] = 0.5f;
            dx2[x] = 0.5f;
        }
    }

    result.y_original = y_original;
    result.x_original = x_original;
    result.input_width_mul_y1 = input_width_mul_y1;
    result.input_width_mul_y2 = input_width_mul_y2;
    result.in_x1 = in_x1;
    result.in_x2 = in_x2;
    result.dy1 = dy1;
    result.dy2 = dy2;
    result.dx1 = dx1;
    result.dx2 = dx2;
    result.batch_size = batch_size;
    result.num_channels = num_channels;
    result.output_height = output_height;
    result.output_width = output_width;
    result.input_height = input_height;
    result.input_width = input_width;

    return result;
}

float InterpolateEvalHelper::get_in_coord(float coord, int64_t axis_idx)
{
    float scale = m_scales[axis_idx];
    int64_t axis = m_axes[axis_idx];
    float length_resized = static_cast<float>(m_out_shape[axis]);
    float length_original = static_cast<float>(m_input_data_shape[axis]);
    return m_get_original_coord(coord, scale, length_resized, length_original);
}

InterpolateEvalHelper::InfoForLinearMode InterpolateEvalHelper::get_info_for_linear_mode()
{
    std::size_t input_rank = m_input_data_shape.size();
    std::size_t num_of_axes = m_axes.size();
    bool is_downsample = false;
    for (std::size_t scale : m_scales)
    {
        is_downsample = is_downsample || (scale < 1.0);
    }

    bool antialias = is_downsample && m_antialias;

    std::vector<float> a(num_of_axes);
    std::vector<int64_t> r(num_of_axes);

    CoordinateTransform output_transform(m_out_shape);
    CoordinateTransform input_transform(m_input_data_shape);

    std::vector<std::size_t> vector_for_indeces(num_of_axes);
    float prod_a = 1;
    for (std::size_t i = 0; i < num_of_axes; ++i)
    {
        a[i] = antialias ? m_scales[i] : 1.0;
        prod_a *= a[i];
        r[i] = (m_scales[i] > 1.0) ? static_cast<int64_t>(2)
                                   : static_cast<int64_t>(std::ceil(2.0f / a[i]));
        vector_for_indeces[i] = 2 * r[i] + 1;
    }
    Shape shape_for_indeces{vector_for_indeces};

    InfoForLinearMode result;

    result.antialias = antialias;
    result.a = a;
    result.r = r;
    result.prod_a = prod_a;
    result.shape_for_indeces = shape_for_indeces;

    return result;
}

InterpolateEvalHelper::ICoords InterpolateEvalHelper::get_icoords(const Coordinate& output_coord)
{
    ICoords result;

    std::size_t input_rank = m_input_data_shape.size();
    std::size_t num_of_axes = m_axes.size();

    std::vector<float> icoords(input_rank);
    std::vector<int64_t> icoords_r(input_rank);
    for (std::size_t i = 0; i < input_rank; ++i)
    {
        icoords[i] = static_cast<float>(output_coord[i]);
        icoords_r[i] = output_coord[i];
    }

    for (std::size_t i = 0; i < num_of_axes; ++i)
    {
        int64_t axis = m_axes[i];
        float coordinate = static_cast<float>(output_coord[axis]);
        float in_coord = get_in_coord(coordinate, i);
        icoords[axis] = in_coord;
        icoords_r[axis] = static_cast<int64_t>(std::round(in_coord));
    }

    result.icoords = icoords;
    result.icoords_r = icoords_r;

    return result;
}

InterpolateEvalHelper::LinearModeInnerIterationResult
    InterpolateEvalHelper::inner_calculation(const Coordinate& output_coord,
                                             const ICoords& icoords_data,
                                             const InfoForLinearMode& info,
                                             const Coordinate& index)
{
    std::size_t input_rank = m_input_data_shape.size();
    std::size_t num_of_axes = m_axes.size();

    LinearModeInnerIterationResult result;

    std::vector<int64_t> inner_coords_vector(input_rank);
    for (std::size_t i = 0; i < input_rank; ++i)
    {
        inner_coords_vector[i] = output_coord[i];
    }

    for (std::size_t i = 0; i < num_of_axes; ++i)
    {
        int64_t axis = m_axes[i];
        inner_coords_vector[axis] = index[i] + icoords_data.icoords_r[axis] - info.r[i];
    }

    bool condition = true;
    for (int64_t axis : m_axes)
    {
        condition = condition && (inner_coords_vector[axis] >= 0) &&
                    (inner_coords_vector[axis] < m_input_data_shape[axis]);
    }

    result.condition = condition;
    if (!condition)
    {
        return result;
    }

    std::vector<float> dz(num_of_axes);
    for (std::size_t i = 0; i < num_of_axes; ++i)
    {
        int64_t axis = m_axes[i];
        dz[i] = icoords_data.icoords[axis] - inner_coords_vector[axis];
    }

    float w = info.prod_a;
    for (std::size_t i = 0; i < num_of_axes; ++i)
    {
        w *= triangle_coeff(info.a[i] * dz[i]);
    }

    std::vector<std::size_t> unsigned_inner_coords_vector(input_rank);
    for (std::size_t i = 0; i < input_rank; ++i)
    {
        unsigned_inner_coords_vector[i] = inner_coords_vector[i];
    }

    Coordinate inner_coord{unsigned_inner_coords_vector};

    result.w = w;
    result.inner_coord = inner_coord;

    return result;
}
