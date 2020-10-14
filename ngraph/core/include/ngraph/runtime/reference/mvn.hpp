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
#include <ngraph/runtime/opt_kernel/broadcast.hpp>
#include <ngraph/runtime/reference/mean.hpp>
#include <ngraph/runtime/reference/multiply.hpp>
#include <ngraph/runtime/reference/sqrt.hpp>
#include <ngraph/runtime/reference/subtract.hpp>
#include <ngraph/runtime/reference/sum.hpp>
#include <ngraph/runtime/reference/autobroadcast_binop.hpp>
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
                std::vector<T> mean_val(shape_size(reduced_shape));
                mean(arg, mean_val.data(), in_shape, reduction_axes, true);
                subtract(arg, mean_val.data(), out, in_shape, reduced_shape, op::AutoBroadcastSpec::NUMPY);

                if (normalize_variance)
                {
                    std::vector<T> multiply_val(shape_size(in_shape));
                    multiply(out, out, multiply_val.data(), shape_size(in_shape));
                    std::vector<T> sum_val(shape_size(reduced_shape));
                    sum(multiply_val.data(), sum_val.data(), in_shape, reduction_axes, true);
                    std::vector<T> broadcast_sum(shape_size(in_shape));
                    broadcast(sum_val.data(),
                              broadcast_sum.data(),
                              reduced_shape,
                              in_shape,
                              reduction_axes);

                    size_t n = 1;
                    for (auto i : reduction_axes)
                    {
                        n *= in_shape[i];
                    }
                    for (size_t i = 0; i < shape_size(in_shape); ++i)
                    {
                        out[i] /= std::sqrt(broadcast_sum[i] / n) + eps;
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
