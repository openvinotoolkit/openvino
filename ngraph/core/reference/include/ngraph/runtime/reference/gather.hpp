// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <numeric>

#include "ngraph/coordinate_range.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/gather_nd.hpp"
#include "utils/span.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename U>
            void gather(const T* const data,
                        const U* const indices,
                        T* out,
                        const Shape& data_shape,
                        const Shape& indices_shape,
                        const Shape& out_shape,
                        size_t axis,
                        size_t batch_dims = 0)
            {
                const auto batch_flattened_shape = shape_size(span(data_shape).subspan(0, batch_dims));
                const auto outer_flattened_size = shape_size(span(data_shape).subspan(batch_dims, axis));
                const auto indices_flattened_shape = shape_size(span(indices_shape).subspan(batch_dims));
                const auto inner_flattened_shape = shape_size(span(data_shape).subspan(axis + 1));
                const auto size_along_axis = data_shape[axis];
                int64_t offset, idx;

                for (int64_t batch_idx = 0; batch_idx < batch_flattened_shape; batch_idx++)
                    for (int64_t outer_idx = 0; outer_idx < outer_flattened_size; outer_idx++) {
                        offset = inner_flattened_shape * size_along_axis * outer_idx;
                        for (int64_t i = 0; i < indices_flattened_shape; i++) {
                            idx = indices[i];
                            if (idx >= size_along_axis  || (idx < 0 && -idx >= size_along_axis))
                                throw std::domain_error{"indices values of Gather exceed size along axis"};
                            if (idx < 0)
                                idx += size_along_axis;

                            const auto src_begin = std::next(data, offset + inner_flattened_shape * idx);
                            const auto src_end = std::next(src_begin, inner_flattened_shape);
//                            const auto out_ = std::next(src_begin, inner_flattened_shape);

                            std::copy(src_begin, src_end, out);
                            out += inner_flattened_shape;
                        }
                    }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
