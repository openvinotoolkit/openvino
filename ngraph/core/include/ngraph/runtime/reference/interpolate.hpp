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

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <list>
#include <map>
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            using Nearest_mode = ngraph::op::v4::Interpolate::NearestMode;
            using Transform_mode = ngraph::op::v4::Interpolate::CoordinateTransformMode;
            using InterpolateMode = ngraph::op::v4::Interpolate::InterpolateMode;

            class GetNearestPixel final
            {
            public:
                GetNearestPixel()
                    : GetNearestPixel(Nearest_mode::round_prefer_floor)
                {
                }

                GetNearestPixel(Nearest_mode mode)
                    : m_mode{mode}
                    , m_func{get_func(mode)}
                {
                }

                ~GetNearestPixel() = default;

                int64_t operator()(float original, bool is_downsample) const
                {
                    return m_func(original, is_downsample);
                }

            private:
                using Func = std::function<int64_t(float, bool)>;

                Nearest_mode m_mode;
                Func m_func;

                Func get_func(Nearest_mode mode)
                {
                    switch (mode)
                    {
                    case Nearest_mode::round_prefer_ceil:
                        return [](float x_original, bool) {
                            return static_cast<int64_t>(std::round(x_original));
                        };
                    case Nearest_mode::floor:
                        return [](float x_original, bool) {
                            return static_cast<int64_t>(std::floor(x_original));
                        };
                    case Nearest_mode::ceil:
                        return [](float x_original, bool) {
                            return static_cast<int64_t>(std::ceil(x_original));
                        };
                    case Nearest_mode::simple:
                        return [](float x_original, bool is_downsample) {
                            if (is_downsample)
                            {
                                return static_cast<int64_t>(std::ceil(x_original));
                            }
                            else
                            {
                                return static_cast<int64_t>(x_original);
                            }
                        };
                    default:;
                    }
                    return [](float x_original, bool) {
                        if (x_original == static_cast<int64_t>(x_original) + 0.5f)
                        {
                            return static_cast<int64_t>(std::floor(x_original));
                        }
                        return static_cast<int64_t>(std::round(x_original));
                    };
                }
            };

            class GetOriginalCoordinate final
            {
            public:
                GetOriginalCoordinate()
                    : GetOriginalCoordinate(Transform_mode::half_pixel)
                {
                }

                GetOriginalCoordinate(Transform_mode mode)
                    : m_mode{mode}
                    , m_func{get_func(mode)}
                {
                }

                ~GetOriginalCoordinate() = default;

                float operator()(float x_resized,
                                 float x_scale,
                                 float length_resized,
                                 float length_original) const
                {
                    return m_func(x_resized, x_scale, length_resized, length_original);
                }

            private:
                using Func = std::function<float(float, float, float, float)>;

                Transform_mode m_mode;
                Func m_func;

                Func get_func(Transform_mode mode)
                {
                    switch (mode)
                    {
                    case Transform_mode::pytorch_half_pixel:
                        return [](float x_resized, float x_scale, float length_resized, float) {
                            return length_resized > 1 ? (x_resized + 0.5f) / x_scale - 0.5f : 0.0f;
                        };
                        break;
                    case Transform_mode::asymmetric:
                        return [](float x_resized, float x_scale, float, float) {
                            return x_resized / x_scale;
                        };
                        break;
                    case Transform_mode::tf_half_pixel_for_nn:
                        return [](float x_resized, float x_scale, float, float) {
                            return (x_resized + 0.5f) / x_scale;
                        };
                        break;
                    case Transform_mode::align_corners:
                        return [](
                            float x_resized, float, float length_resized, float length_original) {
                            return length_resized == 1
                                       ? 0
                                       : x_resized * (length_original - 1) / (length_resized - 1);
                        };
                        break;
                    default:;
                    }
                    return [](float x_resized, float x_scale, float, float) {
                        return ((x_resized + 0.5f) / x_scale) - 0.5f;
                    };
                }
            };

            class InterpolateEvalHelper final
            {
            public:
                InterpolateEvalHelper() = default;

                InterpolateEvalHelper(const op::v4::Interpolate::InterpolateAttrs& attrs,
                                      const Shape& input_data_shape,
                                      const std::vector<int64_t>& axes,
                                      const Shape& out_shape,
                                      const std::vector<float>& scales)
                    : m_get_nearest_pixel{attrs.nearest_mode}
                    , m_get_original_coord{attrs.coordinate_transformation_mode}
                    , m_interp_mode{attrs.mode}
                    , m_antialias{attrs.antialias}
                    , m_cube_coeff{attrs.cube_coeff}
                    , m_input_data_shape{input_data_shape}
                    , m_axes{axes}
                    , m_out_shape{out_shape}
                    , m_scales{scales}
                {
                }

                ~InterpolateEvalHelper() = default;

                float triangle_coeff(float dz);
                std::array<float, 4> get_cubic_coeff(float s, float a);
                float get_in_coord(float coord, int64_t axis_idx);

                Coordinate get_input_coords_for_nearest_mode(const Coordinate& output_coord);

                struct InfoForLinearONNXMode
                {
                    std::vector<float> y_original;
                    std::vector<float> x_original;

                    std::vector<int64_t> input_width_mul_y1;
                    std::vector<int64_t> input_width_mul_y2;
                    std::vector<int64_t> in_x1;
                    std::vector<int64_t> in_x2;

                    std::vector<float> dy1;
                    std::vector<float> dy2;
                    std::vector<float> dx1;
                    std::vector<float> dx2;

                    int64_t batch_size;
                    int64_t num_channels;
                    int64_t output_height;
                    int64_t output_width;
                };

                InfoForLinearONNXMode get_info_for_linear_onnx_mode();

                int64_t clip_coord(int64_t coord, float length)
                {
                    return std::max(static_cast<int64_t>(0),
                                    std::min(coord, static_cast<int64_t>(length) - 1));
                }
            private:
                GetNearestPixel m_get_nearest_pixel;
                GetOriginalCoordinate m_get_original_coord;
                InterpolateMode m_interp_mode;
                double m_cube_coeff;
                bool m_antialias;

                Shape m_input_data_shape;
                std::vector<int64_t> m_axes;
                Shape m_out_shape;

                std::vector<float> m_scales;
            };

            template <typename T>
            class InterpolateEval final
            {
            public:
                InterpolateEval() = default;

                InterpolateEval(const op::v4::Interpolate::InterpolateAttrs& attrs)
                    : m_attrs{attrs}
                    , m_get_nearest_pixel{attrs.nearest_mode}
                    , m_get_original_coord{attrs.coordinate_transformation_mode}
                    , m_interp_mode{attrs.mode}
                    , m_antialias{attrs.antialias}
                    , m_cube_coeff{attrs.cube_coeff}
                {
                }

                ~InterpolateEval() = default;

                void operator()(const T* input_data,
                                const Shape& input_data_shape,
                                const std::vector<float>& scales,
                                const std::vector<int64_t>& axes,
                                T* out,
                                const Shape& out_shape)
                {
                    m_input_data_shape = input_data_shape;
                    m_axes = axes;
                    m_out_shape = out_shape;

                    std::size_t output_data_size = shape_size(out_shape);
                    std::fill(out, out + output_data_size, T{});

                    m_scales = scales;

                    helper =
                        InterpolateEvalHelper{m_attrs, input_data_shape, axes, out_shape, scales};

                    switch (m_interp_mode)
                    {
                    case InterpolateMode::nearest: nearest_func(input_data, out); break;
                    case InterpolateMode::linear: linear_func(input_data, out); break;
                    case InterpolateMode::linear_onnx: linear_onnx_func(input_data, out); break;
                    case InterpolateMode::cubic: cubic_func(input_data, out); break;
                    }
                }

            private:
                op::v4::Interpolate::InterpolateAttrs m_attrs;

                GetNearestPixel m_get_nearest_pixel;
                GetOriginalCoordinate m_get_original_coord;
                InterpolateMode m_interp_mode;
                double m_cube_coeff;
                bool m_antialias;

                Shape m_input_data_shape;
                std::vector<int64_t> m_axes;
                Shape m_out_shape;

                std::vector<float> m_scales;

                InterpolateEvalHelper helper;

                void linear_func(const T* input_data, T* out);
                void linear_onnx_func(const T* input_data, T* out);
                void cubic_func(const T* input_data, T* out);
                void nearest_func(const T* input_data, T* out);
            };

            template <typename T>
            void InterpolateEval<T>::linear_func(const T* input_data, T* out)
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

                for (const Coordinate& output_coord : output_transform)
                {
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
                        float in_coord = helper.get_in_coord(coordinate, i);
                        icoords[axis] = in_coord;
                        icoords_r[axis] = static_cast<int64_t>(std::round(in_coord));
                    }

                    float summa = 0.0f;
                    float wsum = 0.0f;

                    CoordinateTransform indices{shape_for_indeces};
                    for (const auto& index : indices)
                    {
                        std::vector<int64_t> inner_coords_vector(input_rank);
                        for (std::size_t i = 0; i < input_rank; ++i)
                        {
                            inner_coords_vector[i] = output_coord[i];
                        }

                        for (std::size_t i = 0; i < num_of_axes; ++i)
                        {
                            int64_t axis = m_axes[i];
                            inner_coords_vector[axis] = index[i] + icoords_r[axis] - r[i];
                        }

                        bool condition = true;
                        for (int64_t axis : m_axes)
                        {
                            condition = condition && (inner_coords_vector[axis] >= 0) &&
                                        (inner_coords_vector[axis] < m_input_data_shape[axis]);
                        }

                        if (!condition)
                        {
                            continue;
                        }

                        std::vector<float> dz(num_of_axes);
                        for (std::size_t i = 0; i < num_of_axes; ++i)
                        {
                            int64_t axis = m_axes[i];
                            dz[i] = icoords[axis] - inner_coords_vector[axis];
                        }

                        float w = prod_a;
                        for (std::size_t i = 0; i < num_of_axes; ++i)
                        {
                            w *= helper.triangle_coeff(a[i] * dz[i]);
                        }

                        std::vector<std::size_t> unsigned_inner_coords_vector(input_rank);
                        for (std::size_t i = 0; i < input_rank; ++i)
                        {
                            unsigned_inner_coords_vector[i] = inner_coords_vector[i];
                        }

                        Coordinate inner_coord{unsigned_inner_coords_vector};

                        wsum += w;
                        summa +=
                            w * static_cast<float>(input_data[input_transform.index(inner_coord)]);
                    }

                    if (wsum == 0.0f)
                    {
                        out[output_transform.index(output_coord)] = T{};
                    }
                    else
                    {
                        out[output_transform.index(output_coord)] = static_cast<T>(summa / wsum);
                    }
                }
            }

            template <typename T>
            void InterpolateEval<T>::linear_onnx_func(const T* input_data, T* out)
            {
                std::size_t input_rank = m_input_data_shape.size();

                assert((input_rank == 2) || (input_rank == 4));
                assert(m_axes.size() == 2);
                bool correct_axes = ((m_axes[0] == 0) && (m_axes[1] == 1)) ||
                                    ((m_axes[0] == 2) && (m_axes[1] == 3));
                assert(correct_axes);

                const auto info = helper.get_info_for_linear_onnx_mode();

                int64_t batch_size = info.batch_size;
                int64_t num_channels = info.num_channels;
                int64_t output_height = info.output_height;
                int64_t output_width = info.output_width;

                const T* xdata = input_data;
                T* ydata = out;
                for (int64_t n = 0; n < batch_size; ++n)
                {
                    for (int64_t c = 0; c < num_channels; ++c)
                    {
                        for (int64_t y = 0; y < output_height; ++y)
                        {
                            for (int64_t x = 0; x < output_width; ++x)
                            {
                                T x11 = xdata[info.input_width_mul_y1[y] + info.in_x1[x]];
                                T x21 = xdata[info.input_width_mul_y1[y] + info.in_x2[x]];
                                T x12 = xdata[info.input_width_mul_y2[y] + info.in_x1[x]];
                                T x22 = xdata[info.input_width_mul_y2[y] + info.in_x2[x]];

                                ydata[output_width * y + x] =
                                    static_cast<T>(info.dx2[x] * info.dy2[y] * x11 +
                                                   info.dx1[x] * info.dy2[y] * x21 +
                                                   info.dx2[x] * info.dy1[y] * x12 +
                                                   info.dx1[x] * info.dy1[y] * x22);
                            }
                        }
                        xdata += input_height * input_width;
                        ydata += output_width * output_height;
                    }
                }
            }

            template <typename T>
            void InterpolateEval<T>::cubic_func(const T* input_data, T* out)
            {
                std::size_t input_rank = m_input_data_shape.size();
                std::size_t num_of_axes = m_axes.size();

                CoordinateTransform output_transform(m_out_shape);
                CoordinateTransform input_transform(m_input_data_shape);
                Shape indices_shape{std::vector<std::size_t>(num_of_axes, 4)};

                for (const Coordinate& output_coord : output_transform)
                {
                    auto input_coord = output_coord;
                    std::map<std::size_t, std::array<float, 4>> cubic_coeffs;
                    for (std::size_t i = 0; i < num_of_axes; ++i)
                    {
                        int64_t axis = m_axes[i];
                        float coordinate = static_cast<float>(output_coord[axis]);
                        float in_coord = helper.get_in_coord(coordinate, i);
                        int64_t in_coord_int = static_cast<int64_t>(std::floor(in_coord));
                        input_coord[axis] = in_coord_int;
                        auto s = static_cast<float>(in_coord - in_coord_int);
                        cubic_coeffs[axis] = helper.get_cubic_coeff(s, m_cube_coeff);
                    }

                    float summa = 0.0f;
                    CoordinateTransform indices{indices_shape};
                    for (const Coordinate& idx : indices)
                    {
                        auto coords_for_sum = input_coord;
                        float coeffs_prod = 1.0;
                        for (std::size_t i = 0; i < num_of_axes; ++i)
                        {
                            int64_t axis = m_axes[i];
                            coords_for_sum[axis] =
                                helper.clip_coord(input_coord[axis] + idx[i] - 1,
                                                  static_cast<float>(m_input_data_shape[axis]));
                            coeffs_prod *= cubic_coeffs[axis][idx[i]];
                        }
                        summa += coeffs_prod * input_data[input_transform.index(coords_for_sum)];
                    }

                    out[output_transform.index(output_coord)] = static_cast<T>(summa);
                }
            }

            template <typename T>
            void InterpolateEval<T>::nearest_func(const T* input_data, T* out)
            {
                CoordinateTransform output_transform(m_out_shape);
                CoordinateTransform input_transform(m_input_data_shape);

                for (const Coordinate& output_coord : output_transform)
                {
                    auto input_coord = get_input_coords_for_nearest_mode(output_coord);
                    out[output_transform.index(output_coord)] =
                        input_data[input_transform.index(input_coord)];
                }
            }

            template <typename T>
            void interpolate(const T* input_data,
                             const Shape& input_data_shape,
                             const std::vector<float>& scales,
                             const std::vector<int64_t>& axes,
                             T* out,
                             const Shape& out_shape,
                             const op::v4::Interpolate::InterpolateAttrs& attrs)
            {
                InterpolateEval<T> evaluator{attrs};
                evaluator(input_data, input_data_shape, scales, axes, out, out_shape);
            }
        }
    }
}
