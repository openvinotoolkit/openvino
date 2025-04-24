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

    const Shape batch_shape(begin(params_shape), next(begin(params_shape), batch_dims));
    const auto batch_size = shape_size(batch_shape);

    if (!std::equal(begin(params_shape), next(begin(params_shape), batch_dims), begin(indices_shape))) {
        throw std::domain_error{"dimensions in params and indices have to be equal on batch dimensions"};
    }

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

    for (size_t batch = 0; batch != batch_size; ++batch) {
        const auto input_batch_offset = batch * batch_offset;
        const auto output_batch_offset = batch * number_of_slices_to_copy_in_one_batch * slice_size;
        const auto coordinates_batch_offset = batch * number_of_slices_to_copy_in_one_batch * coordinates_size;
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
