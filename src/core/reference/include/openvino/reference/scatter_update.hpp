// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <numeric>

#include "openvino/core/shape.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace reference {
static const CoordinateTransformBasic get_target_shape(const Shape& data_shape,
                                                       const Coordinate& start_corner,
                                                       const Coordinate& end_corner) {
    const auto m_n_axes = data_shape.size();
    Shape target_shape;
    target_shape.reserve(m_n_axes);
    AxisVector axis_order(m_n_axes);
    std::iota(axis_order.begin(), axis_order.end(), 0);
    const Strides strides(m_n_axes, 1);
    for (size_t axis = 0; axis < m_n_axes; axis++) {
        target_shape.push_back(
            util::ceil_div(end_corner[axis_order[axis]] - start_corner[axis_order[axis]], strides[axis_order[axis]]));
    }
    return target_shape;
}

static void scatter_update(const char* input_data,
                           const int64_t* indices,
                           const char* updates,
                           const int64_t axis,
                           char* out_buf,
                           const size_t elem_size,
                           const Shape& data_shape,
                           const Shape& indices_shape,
                           const Shape& updates_shape) {
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
    CoordinateTransformBasic indices_transform{indices_shape};
    const auto indices_in_strides = row_major_strides(indices_shape);

    size_t indices_ndim = indices_shape.size();
    size_t updates_ndim = updates_shape.size();
    size_t data_ndim = data_shape.size();

    const auto size_after_axis = shape_size(Shape(data_shape.begin() + axis + 1, data_shape.end()));
    size_t num_axis_jumps{0};
    int num_unary_moves{0};
    for (size_t i = axis + 1; i < updates_ndim; ++i) {
        const auto updates_size_after_axis = shape_size(Shape(updates_shape.begin() + i, updates_shape.end()));
        if (updates_size_after_axis > size_after_axis)
            ++num_axis_jumps;
        if (updates_shape[i] == 1)
            ++num_unary_moves;
    }

    if (!num_axis_jumps)
        num_axis_jumps = updates_ndim - data_ndim;

    auto updates_axis_dim = static_cast<size_t>(axis + num_axis_jumps + num_unary_moves);

    if (updates_axis_dim >= updates_ndim)
        updates_axis_dim = updates_ndim - 1;

    Coordinate updates_indices_start_corner(updates_ndim, 0);
    Coordinate updates_indices_end_corner(updates_ndim, 1);

    for (size_t i = 0; i < indices_ndim; ++i) {
        updates_indices_end_corner[axis + i] = updates_shape[axis + i];
    }

    const auto updates_indices_transform =
        get_target_shape(updates_shape, updates_indices_start_corner, updates_indices_end_corner);
    auto updates_indices_coord_iter = updates_indices_transform.begin();

    int iteration{0};
    for (const Coordinate& indices_cord : indices_transform) {
        const size_t indices_idx =
            std::inner_product(indices_cord.begin(), indices_cord.end(), indices_in_strides.begin(), uint64_t(0));
        int64_t slice_index = indices[indices_idx];

        Coordinate out_start_corner(data_shape.size(), 0);
        Coordinate out_end_corner(data_shape);
        out_start_corner[axis] = static_cast<size_t>(slice_index);
        out_end_corner[axis] = out_start_corner[axis] + 1;

        const auto out_transform = get_target_shape(data_shape, out_start_corner, out_end_corner);
        const auto out_transform_in_strides = row_major_strides(data_shape);

        if (updates_indices_coord_iter == updates_indices_transform.end())
            break;
        Coordinate updates_update_start_corner = *updates_indices_coord_iter;
        Coordinate updates_update_end_corner(updates_shape);
        for (size_t i = 0; i < indices_ndim; ++i) {
            updates_update_end_corner[axis + i] = updates_update_start_corner[axis + i] + 1;
        }

        const auto updates_update_transform =
            get_target_shape(updates_shape, updates_update_start_corner, updates_update_end_corner);
        const auto updates_update_in_strides = row_major_strides(updates_shape);
        auto updates_update_coord_iter = updates_update_transform.begin();

        for (const Coordinate& out_cord : out_transform) {
            if (updates_update_coord_iter == updates_update_transform.end())
                break;
            Coordinate update_cord = *updates_update_coord_iter;
            Coordinate out_coord = out_cord;
            out_coord.at(axis) = slice_index;
            update_cord.at(updates_axis_dim) += iteration;
            const auto data_idx =
                std::inner_product(out_coord.begin(), out_coord.end(), out_transform_in_strides.begin(), uint64_t(0));
            const auto updates_idx = std::inner_product(update_cord.begin(),
                                                        update_cord.end(),
                                                        updates_update_in_strides.begin(),
                                                        uint64_t(0)) *
                                     elem_size;

            std::copy(updates + updates_idx, updates + (updates_idx + elem_size), out_buf + data_idx * elem_size);
            updates_update_coord_iter++;
        }
        updates_indices_coord_iter++;
        iteration++;
    }
}
}  // namespace reference
}  // namespace ov
