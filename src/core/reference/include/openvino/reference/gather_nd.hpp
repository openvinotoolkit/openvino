// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>

#include "openvino/reference/utils/coordinate_transform.hpp"
#include "utils/span.hpp"

namespace ov {
namespace reference {
namespace details {
template <typename Iterator>
std::vector<size_t> get_indices_offsets(const Iterator beg, const Iterator end, size_t last_slice_size) {
    auto next_e = beg;
    auto i = std::distance(beg, end);
    std::vector<size_t> offsets(i + 1, last_slice_size);
    while (i-- > 0) {
        offsets[i] = *next_e * offsets[i + 1];
        ++next_e;
    }

    return offsets;
}
}  // namespace details

///
/// Implementation find maximum length of *slice* of input *params* which might be
/// copied to *out* index by index.
/// +-------+--------------+-------+
/// | batch | indices[:-1] | slice |
/// | shape |   shape      | shape |
/// +-------+--------------+-------+
///
template <typename T, typename U>
void gather_nd(const T* const params,
               const U* const indices,
               T* const out,
               const Shape& params_shape,
               const Shape& indices_shape,
               const Shape& out_shape,
               const int batch_dims = 0) {
    using std::begin;
    using std::end;
    using std::next;
    using std::prev;
    const auto rbegin = [](const Shape& s) {  // generic since C++14
        return s.rbegin();
    };

    // Check broadcastability and compute output batch shape
    Shape batch_shape;
    batch_shape.reserve(batch_dims);
    for (int i = 0; i < batch_dims; ++i) {
        const auto params_dim = params_shape[i];
        const auto indices_dim = indices_shape[i];
        if (params_dim == indices_dim) {
            batch_shape.push_back(params_dim);
        } else if (params_dim == 1) {
            batch_shape.push_back(indices_dim);  // broadcast params dimension
        } else if (indices_dim == 1) {
            batch_shape.push_back(params_dim);  // broadcast indices dimension
        } else {
            throw std::domain_error{"dimensions in params and indices have to be broadcastable on batch dimensions"};
        }
    }
    const auto batch_size = shape_size(batch_shape);

    const auto first_slice_index_in_params = batch_dims + indices_shape.back();

    if (!(first_slice_index_in_params <= params_shape.size())) {
        throw std::domain_error{"params_shape should have enough rank to be index by indices"};
    }

    const auto slice_shape = span(params_shape).subspan(first_slice_index_in_params);
    const auto slice_size = shape_size(slice_shape);

    const auto dims_begin = next(rbegin(params_shape), slice_shape.size());
    const auto dims_end = next(dims_begin, indices_shape.back() - 1);

    const auto indices_offsets = details::get_indices_offsets(dims_begin, dims_end, slice_size);

    const auto batch_offset = indices_offsets.front() * params_shape[batch_dims];

    const auto k_1_indices = span(next(begin(indices_shape), batch_dims), prev(end(indices_shape)));

    const auto k_1_params = span(next(begin(params_shape), batch_dims), prev(end(params_shape)));

    const auto number_of_slices_to_copy_in_one_batch = shape_size(k_1_indices);

    const auto coordinates_size = indices_shape.back();

    // Compute strides for params and indices batch dimensions
    std::vector<size_t> params_batch_strides(batch_dims);
    std::vector<size_t> indices_batch_strides(batch_dims);
    if (batch_dims > 0) {
        // Calculate strides from right to left (C-order)
        params_batch_strides[batch_dims - 1] = batch_offset;
        indices_batch_strides[batch_dims - 1] = number_of_slices_to_copy_in_one_batch * coordinates_size;

        for (int i = batch_dims - 2; i >= 0; --i) {
            params_batch_strides[i] = params_batch_strides[i + 1] * params_shape[i + 1];
            indices_batch_strides[i] = indices_batch_strides[i + 1] * indices_shape[i + 1];
        }
    }

    for (size_t batch = 0; batch != batch_size; ++batch) {
        // Compute actual offsets considering broadcasting
        size_t input_batch_offset = 0;
        size_t coordinates_batch_offset = 0;
        size_t batch_idx = batch;

        // Convert linear batch index to multi-dimensional and compute offsets
        for (int i = batch_dims - 1; i >= 0; --i) {
            const auto dim_idx = batch_idx % batch_shape[i];
            batch_idx /= batch_shape[i];

            // If dimension is 1 (broadcast), use index 0; otherwise use dim_idx
            const auto params_idx = (params_shape[i] == 1) ? 0 : dim_idx;
            const auto indices_idx = (indices_shape[i] == 1) ? 0 : dim_idx;

            input_batch_offset += params_idx * params_batch_strides[i];
            coordinates_batch_offset += indices_idx * indices_batch_strides[i];
        }

        const auto output_batch_offset = batch * number_of_slices_to_copy_in_one_batch * slice_size;
        for (size_t slice = 0; slice != number_of_slices_to_copy_in_one_batch; ++slice) {
            const auto slice_coordinates = next(indices, coordinates_batch_offset + slice * coordinates_size);

            size_t input_slice_offset = input_batch_offset;
            for (size_t c = 0; c != coordinates_size; ++c) {
                const auto i_c = slice_coordinates[c];
                const auto index = i_c < 0 ? k_1_params[c] + i_c : i_c;
                input_slice_offset += index * indices_offsets[c];
            }
            const auto output_slice_offset = output_batch_offset + slice * slice_size;
            std::copy(next(params, input_slice_offset),
                      next(params, input_slice_offset + slice_size),
                      next(out, output_slice_offset));
        }
    }
}

}  // namespace reference
}  // namespace ov
