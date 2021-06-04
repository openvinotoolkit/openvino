// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include "ngraph/runtime/reference/matmul.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            namespace details
            {
                std::vector<size_t> get_transpose_order(const Shape& input_shape)
                {
                    size_t rank = input_shape.size();
                    NGRAPH_CHECK(rank > 1, "Invalid input for transpose");
                    std::vector<size_t> axes_order(rank);
                    std::iota(axes_order.begin(), axes_order.end(), 0);
                    std::swap(axes_order[rank - 1], axes_order[rank - 2]);
                    return axes_order;
                }
            } // namespace details
        }     // namespace reference
    }         // namespace runtime
} // namespace ngraph
