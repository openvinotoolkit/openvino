// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>
#include <cmath>
#include <numeric>
#include <vector>

#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void transpose(const char* data,
                           char* out,
                           const Shape& data_shape,
                           size_t element_size,
                           const int64_t* axes_order,
                           Shape out_shape)
            {
                // To reuse opt_kernel::reshape axes order vector has to be converted to AxisVector
                // Negative axes are not supported, it is validated by transpose evaluate method
                std::vector<size_t> axis_vector(axes_order, axes_order + data_shape.size());
                runtime::opt_kernel::reshape(
                    data, out, data_shape, axis_vector, out_shape, element_size);
            }

            // Legacy function template to ensure backward compatibility
            // Can be removed after ARM plugin start using evaluate or no template function
            template <typename T, typename U>
            NGRAPH_DEPRECATED(
                "Traspose function with template types is deprecated, use function with char* "
                "args.")
            void transpose(const T* arg, T* out, Shape arg_shape, const U* axes_order = nullptr)
            {
                std::vector<std::int64_t> converted_axes_order(arg_shape.size());
                if (axes_order == nullptr)
                {
                    std::iota(converted_axes_order.begin(), converted_axes_order.end(), 0);
                    std::reverse(converted_axes_order.begin(), converted_axes_order.end());
                }
                else
                {
                    for (size_t i = 0; i < converted_axes_order.size(); ++i)
                    {
                        converted_axes_order[i] = static_cast<std::int64_t>(axes_order[i]);
                    }
                }
                Shape output_shape(arg_shape.size());
                std::transform(
                    converted_axes_order.begin(),
                    converted_axes_order.end(),
                    output_shape.begin(),
                    [&](const int64_t& v) {
                        NGRAPH_CHECK(v >= 0,
                                     "Negative values for transpose axes order are not supported.");
                        NGRAPH_CHECK(v < int64_t(arg_shape.size()),
                                     "Transpose axis ",
                                     v,
                                     " is out of shape range.");
                        return arg_shape[v];
                    });

                transpose(reinterpret_cast<const char*>(arg),
                          reinterpret_cast<char*>(out),
                          arg_shape,
                          sizeof(T),
                          converted_axes_order.data(),
                          output_shape);
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
