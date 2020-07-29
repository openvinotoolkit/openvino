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

#include <cmath>
#include <cstddef>
#include <functional>
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            using Nearest_mode = op::v4::Interpolate::NearestMode;
            using Transform_mode = ngraph::op::v4::Interpolate::CoordinateTransformMode;
            using InterpolateMode = op::v4::Interpolate::InterpolateMode;

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
                    case Nearest_mode::round_prefer_floor:
                        return [](float x_original, bool) {
                            if (x_original == static_cast<int64_t>(x_original) + 0.5f)
                            {
                                return static_cast<int64_t>(std::floor(x_original));
                            }
                            return static_cast<int64_t>(std::round(x_original));
                        };
                    case Nearest_mode::round_prefer_ceil:
                        return [](float x_original, bool) {
                            return static_cast<int64_t>(std::round(x_original));
                        };
                    case Nearest_mode::floor:
                        return [](float x_original, bool) {
                            return static_cast<int64_t>(std::floor(x_original));
                        };
                        break;
                    case Nearest_mode::ceil:
                        return [](float x_original, bool) {
                            return static_cast<int64_t>(std::ceil(x_original));
                        };
                        break;
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
                        break;
                    }
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
                    case Transform_mode::half_pixel:
                        return [](float x_resized, float x_scale, float, float) {
                            return ((x_resized + 0.5f) / x_scale) - 0.5f;
                        };
                        break;
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
                    }
                }
            };

            template <typename T>
            class InterpolateEval final
            {
            public:
                InterpolateEval() = default;

                InterpolateEval(const op::v4::Interpolate::InterpolateAttrs& attrs)
                    : m_get_nearest_pixel{attrs.nearest_mode}
                    , m_get_original_coord{attrs.coordinate_transformation_mode}
                    , m_interp_mode{attrs.mode}
                    , m_pads_begin{attrs.pads_begin}
                    , m_pads_end{attrs.pads_end}
                    , m_antialias{attrs.antialias}
                    , m_cube_coeff{attrs.cube_coeff}
                {
                }

                ~InterpolateEval() = default;

                bool operator()(const T* input_data,
                                const Shape& input_data_shape,
                                const std::vector<int64_t>& target_spatial_shape,
                                const std::vector<int64_t>& axes,
                                T* out,
                                const Shape& out_shape);

            private:
                GetNearestPixel m_get_nearest_pixel;
                GetOriginalCoordinate m_get_original_coord;
                InterpolateMode m_interp_mode;
                std::vector<size_t> m_pads_begin;
                std::vector<size_t> m_pads_end;
                double m_cube_coeff;
                bool m_antialias;
            };

            template <typename T>
            void interpolate(const T* input_data,
                             const Shape& input_data_shape,
                             const std::vector<int64_t>& target_spatial_shape,
                             const std::vector<int64_t>& axes,
                             T* out,
                             const Shape& out_shape,
                             const op::v4::Interpolate::InterpolateAttrs& attrs)
            {
                InterpolateEval evaluator{attrs};
            }
        }
    }
}
