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
#include <ngraph/runtime/reference/add.hpp>
#include <ngraph/runtime/reference/divide.hpp>
#include <ngraph/runtime/reference/mean.hpp>
#include <ngraph/runtime/reference/multiply.hpp>
#include <ngraph/runtime/reference/sqrt.hpp>
#include <ngraph/runtime/reference/subtract.hpp>
#include <ngraph/runtime/reference/sum.hpp>
#include <ngraph/shape.hpp>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void mvn(const T* arg,
                     T* out,
                     const Shape& in_shape,
                     bool normalize_variance,
                     AxisSet reduction_axes,
                     double eps)
            {
                auto reduced_shape = reduce(in_shape, reduction_axes, true);
                std::vector<T> tmp_buffer(shape_size(in_shape));
                mean(arg, tmp_buffer.data(), in_shape, reduction_axes, true);
                subtract(arg,
                         tmp_buffer.data(),
                         out,
                         in_shape,
                         reduced_shape,
                         op::AutoBroadcastSpec::NUMPY);

                if (normalize_variance)
                {
                    multiply(out, out, tmp_buffer.data(), shape_size(in_shape));
                    std::vector<T> mean_value(shape_size(reduced_shape));
                    mean(tmp_buffer.data(), mean_value.data(), in_shape, reduction_axes, true);

                    add(mean_value.data(),
                        std::vector<T>(shape_size(reduced_shape), eps).data(),
                        tmp_buffer.data(),
                        reduced_shape,
                        reduced_shape,
                        op::AutoBroadcastSpec::NUMPY);
                    sqrt(tmp_buffer.data(), tmp_buffer.data(), shape_size(reduced_shape));

                    divide(out,
                           tmp_buffer.data(),
                           out,
                           in_shape,
                           reduced_shape,
                           op::AutoBroadcastSpec::NUMPY,
                           true);
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
