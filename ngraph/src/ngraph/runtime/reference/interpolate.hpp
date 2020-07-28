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

#include <cstddef>

#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
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
