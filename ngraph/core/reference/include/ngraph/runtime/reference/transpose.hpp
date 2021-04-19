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
                           const int64_t* axes_order = nullptr,
                           Shape out_shape = {})
            {
                std::vector<size_t> in_axes_order(data_shape.size());
                if (axes_order == nullptr)
                {
                    std::iota(in_axes_order.begin(), in_axes_order.end(), 0);
                    std::reverse(in_axes_order.begin(), in_axes_order.end());
                }
                else
                {
                    std::vector<int64_t> dims(axes_order, axes_order + data_shape.size());
                    std::transform(
                        dims.begin(), dims.end(), in_axes_order.begin(), [&](const int64_t v) {
                            NGRAPH_CHECK(
                                v >= 0,
                                "Negative values for transpose axes order are not supported.");
                            return v;
                        });
                }

                if (out_shape.empty())
                {
                    out_shape.resize(data_shape.size());
                    std::transform(in_axes_order.begin(),
                                   in_axes_order.end(),
                                   out_shape.begin(),
                                   [&](const size_t& v) { return data_shape[v]; });
                }

                runtime::opt_kernel::reshape(
                    data, out, data_shape, in_axes_order, out_shape, element_size);
            }

            // Legacy function template to ensure backward compatibility
            // Can be removed after ARM plugin start using evaluate or no template function
            template <typename T, typename U>
            void transpose(const T* arg, T* out, Shape arg_size, const U* axes_order = nullptr)
            {
                std::vector<std::int64_t> converted_indices(arg_size.size());

                if (axes_order != nullptr)
                {
                    for (size_t i = 0; i < converted_indices.size(); ++i)
                    {
                        converted_indices[i] = static_cast<std::int64_t>(axes_order[i]);
                    }
                    transpose(reinterpret_cast<const char*>(arg),
                              reinterpret_cast<char*>(out),
                              arg_size,
                              sizeof(T),
                              converted_indices.data());
                }
                else
                {
                    transpose(reinterpret_cast<const char*>(arg),
                              reinterpret_cast<char*>(out),
                              arg_size,
                              sizeof(T));
                }
            }
        }
    }
}
