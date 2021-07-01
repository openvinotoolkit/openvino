// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void scatter_update(const char* input_data,
                                const int64_t* indices,
                                const char* updates,
                                const int64_t axis,
                                char* out_buf,
                                const size_t elem_size,
                                const Shape& data_shape,
                                const Shape& indices_shape,
                                const Shape& updates_shape)
            {
                // Copy inputs to out
                std::memcpy(out_buf, input_data, elem_size * shape_size(data_shape));

                // Algorithm overview
                // data[..., indices[m, n, ..., p], ...] = updates[..., m, n, ..., p, ...]
                // where first ... in the data corresponds to first axis dimensions,
                // last ... in the data corresponds to the rank(data) - (axis + 1) dimensions.

                //
                // for i_coord in indices[m, n, ..., p]:
                //     # get linear index
                //     i_idx = index(i_coord)
                //     # simultaneously iterate over two slices of data with same elements count
                //     for d_coord in slice data[..., i_idx, ...],
                //         u_coord in slice updates[..., i_coord, ...]
                //          data[index(d_coord)] = updates[index(u_coord)]

                CoordinateTransform indices_transform{indices_shape};
                CoordinateTransform data_transform{data_shape};

                size_t indices_ndim = indices_shape.size();
                size_t updates_ndim = updates_shape.size();

                // Create an outer CoordinateTransform for "update", which would allow to
                // iterate only over "indices" dimensions:
                // set to "1" all non-indices dimensions
                // updates[1, ..., 1, m, n, ..., p, 1, 1,..., 1]
                Coordinate updates_indices_start_corner(updates_ndim, 0);
                Coordinate updates_indices_end_corner(updates_ndim, 1);
                for (size_t i = 0; i < indices_ndim; ++i)
                {
                    updates_indices_end_corner[axis + i] = updates_shape[axis + i];
                }
                CoordinateTransform updates_indices_transform(
                    updates_shape, updates_indices_start_corner, updates_indices_end_corner);
                // Is needed to simultaneously iterate over updates coordinates while
                // iterating over indices.
                auto updates_indices_coord_iter = updates_indices_transform.begin();

                for (const Coordinate& indices_cord : indices_transform)
                {
                    const size_t indices_idx = indices_transform.index(indices_cord);
                    int64_t slice_index = indices[indices_idx];

                    // Define the extent of coordinates which will be updated.
                    Coordinate out_start_corner(data_shape.size(), 0);
                    Coordinate out_end_corner(data_shape);
                    out_start_corner[axis] = static_cast<size_t>(slice_index);
                    out_end_corner[axis] = out_start_corner[axis] + 1;
                    CoordinateTransform out_transform(data_shape, out_start_corner, out_end_corner);

                    // Define the CoordinateTransform for updates coordinates.
                    // All except indices-dimensions.
                    if (updates_indices_coord_iter == updates_indices_transform.end())
                        break;
                    Coordinate updates_update_start_corner = *updates_indices_coord_iter;
                    Coordinate updates_update_end_corner(updates_shape);
                    for (size_t i = 0; i < indices_ndim; ++i)
                    {
                        updates_update_end_corner[axis + i] =
                            updates_update_start_corner[axis + i] + 1;
                    }
                    // The m, n, .., p symbols stand for values at those axes.
                    // The m+1 means value at axis m plus 1.
                    // udpates_shape (start): [ 0, ...,  0, m  , n  , ... p  ,  0, ...,  0]
                    // updates_shape (end):   [-1, ..., -1, m+1, n+1, ... p+1, -1, ..., -1]
                    CoordinateTransform updates_update_transform(
                        updates_shape, updates_update_start_corner, updates_update_end_corner);
                    auto updates_update_coord_iter = updates_update_transform.begin();
                    for (const Coordinate& out_cord : out_transform)
                    {
                        if (updates_update_coord_iter == updates_update_transform.end())
                            break;
                        const auto src_idx =
                            updates_update_transform.index(*updates_update_coord_iter) * elem_size;
                        std::copy(updates + src_idx,
                                  updates + (src_idx + elem_size),
                                  out_buf + out_transform.index(out_cord) * elem_size);
                        updates_update_coord_iter++;
                    }
                    updates_indices_coord_iter++;
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
