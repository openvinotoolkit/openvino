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

            class GetNearestPixel final
            {
            public:
                GetNearestPixel() :
                    GetNearestPixel(Nearest_mode::round_prefer_floor)
                {
                }

                GetNearestPixel(Nearest_mode mode) :
                    m_mode{mode}, m_func{get_func(mode)}
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
                        return [](float x_original, bool){
                            if (x_original == static_cast<int64_t>(x_original) + 0.5f)
                            {
                                return static_cast<int64_t>(std::floor(x_original));
                            }
                            return static_cast<int64_t>(std::round(x_original));
                        };
                    case round_prefer_ceil:
                        return [](float x_original, bool){
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
                        return  [](float x_original, bool is_downsample) {
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

            class GetOriginalCoordinate{};

            template <typename T>
            void nearest_interpolate(const T* input_data,
                                     const T* target_spatial_shape,
                                     const T* axes,
                                     T* out,
                                     const Shape& input_data_shape,
                                     const Shape& target_spatial_shape_size,
                                     const Shape& axes_size,
                                     const op::v4::Interpolate::InterpolateAttrs& attrs)
            {
            }

            template <typename T>
            void linear_interpolate(const T* input_data,
                                    const T* target_spatial_shape,
                                    const T* axes,
                                    T* out,
                                    const Shape& input_data_shape,
                                    const Shape& target_spatial_shape_size,
                                    const Shape& axes_size,
                                    const op::v4::Interpolate::InterpolateAttrs& attrs)
            {
            }

            template <typename T>
            void linear_onnx_interpolate(const T* input_data,
                                         const T* target_spatial_shape,
                                         const T* axes,
                                         T* out,
                                         const Shape& input_data_shape,
                                         const Shape& target_spatial_shape_size,
                                         const Shape& axes_size,
                                         const op::v4::Interpolate::InterpolateAttrs& attrs)
            {
            }

            template <typename T>
            void cubic_interpolate(const T* input_data,
                                   const T* target_spatial_shape,
                                   const T* axes,
                                   T* out,
                                   const Shape& input_data_shape,
                                   const Shape& target_spatial_shape_size,
                                   const Shape& axes_size,
                                   const op::v4::Interpolate::InterpolateAttrs& attrs)
            {
            }

            template <typename T>
            void interpolate(const T* input_data,
                             const T* target_spatial_shape,
                             const T* axes,
                             T* out,
                             const Shape& input_data_shape,
                             const Shape& target_spatial_shape_size,
                             const Shape& axes_size,
                             const op::v4::Interpolate::InterpolateAttrs& attrs)
            {
                switch (attrs.mode)
                {
                case op::v4::Interpolate::InterpolateMode::nearest:
                    nearest_interpolate<T>(input_data,
                                           target_spatial_shape,
                                           axes,
                                           out,
                                           input_data_shape,
                                           target_spatial_shape_size,
                                           axes_size,
                                           attrs);
                    break;
                case op::v4::Interpolate::InterpolateMode::linear:
                    linear_interpolate<T>(input_data,
                                          target_spatial_shape,
                                          axes,
                                          out,
                                          input_data_shape,
                                          target_spatial_shape_size,
                                          axes_size,
                                          attrs);
                    break;
                case op::v4::Interpolate::InterpolateMode::linear_onnx:
                    linear_onnx_interpolate<T>(input_data,
                                               target_spatial_shape,
                                               axes,
                                               out,
                                               input_data_shape,
                                               target_spatial_shape_size,
                                               axes_size,
                                               attrs);
                    break;
                case op::v4::Interpolate::InterpolateMode::cubic:
                    cubic_interpolate<T>(input_data,
                                         target_spatial_shape,
                                         axes,
                                         out,
                                         input_data_shape,
                                         target_spatial_shape_size,
                                         axes_size,
                                         attrs);
                    break;
                }
            }
        }
    }
}
