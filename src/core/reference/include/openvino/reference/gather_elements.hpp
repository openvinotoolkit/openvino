// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
template <typename T, typename U>
void gather_elements(const T* data,
                     const U* indices,
                     T* out,
                     const Shape& data_shape,
                     const Shape& indices_shape,
                     const Shape& out_shape,
                     int64_t axis) {
    if (axis < 0) {
        axis += data_shape.size();
    }
    if (axis < 0 || axis >= static_cast<int64_t>(data_shape.size())) {
        throw std::domain_error{"axis for GatherElements exceeds allowed range [0, data_rank)"};
    }

    // in 1D case results can be achieved without additional calculations
    if (data_shape.size() == 1) {
        for (size_t i = 0; i < indices_shape[0]; i++) {
            if (static_cast<size_t>(indices[i]) >= data_shape[0]) {
                throw std::domain_error{"indices values of GatherElement exceed data size"};
            }
            out[i] = data[indices[i]];
        }
        return;
    }

    // 2D case is most frequent in order to run faster simpler separate solution
    // implemented
    size_t num_rows = indices_shape[0];
    size_t num_columns = indices_shape[1];
    size_t data_num_columns = data_shape[1];
    if (data_shape.size() == 2) {
        size_t idx;
        if (axis == 0) {
            for (size_t i = 0; i < num_rows; i++)
                for (size_t j = 0; j < num_columns; j++) {
                    idx = indices[num_columns * i + j];
                    if (idx < 0 || idx >= data_shape[0]) {
                        throw std::domain_error{"indices values of GatherElement exceed data size"};
                    }
                    out[num_columns * i + j] = data[data_num_columns * idx + j];
                }
            return;
        } else  // axis == 1
        {
            for (size_t i = 0; i < num_rows; i++)
                for (size_t j = 0; j < num_columns; j++) {
                    idx = indices[num_columns * i + j];
                    if (idx < 0 || idx >= data_shape[1]) {
                        throw std::domain_error{"indices values of GatherElement exceed data size"};
                    }

                    out[num_columns * i + j] = data[data_num_columns * i + idx];
                }
            return;
        }
    }

    /*
     assume data and indices are 5D and axis = 2
     size of indices(N0,N1,N2,N3,N4)
     size of data (N0,N1,N2',N3,N4)

     the offset for indices will be
     N4*N3*N2*N1*n0 + N4*N3*N2*n1 + N4*N3*n2 + N4*n3 + n4
     and for data
     N4*N3*N2'*N1*n0 + N4*N3*N2'*n1 + N4*N3*n2' + N4*n3 + n4
     all values (except n2') are fixed or gradually increase
     most of offset calculations are shared. We can rewrite offset for data as follows

     data_offset = N4*N3*N2'(N1*n0 + n1) + N4*N3*n2' + (N4*n3 + n4)
     N4*N3*N2' - outer_sum_inc
     N4*N3*N2'(N1*n0 + n1) - outer_sum
     N4*N3*n2' - n2' is red from indices tensor n2' = indices[n0,n1,n2,n3,n4]
     (N4*n3 + n4) - inner_sum
    */

    size_t max_inner_sum = 1;
    for (size_t i = axis + 1; i < indices_shape.size(); i++)
        max_inner_sum *= indices_shape[i];

    size_t max_outer_sum = 1, outer_sum_inc = 1;
    for (int i = 0; i < axis; i++)
        max_outer_sum *= indices_shape[i];
    for (size_t i = axis; i < data_shape.size(); i++)
        outer_sum_inc *= data_shape[i];
    max_outer_sum *= outer_sum_inc;

    for (size_t outer_sum = 0, i = 0; outer_sum < max_outer_sum; outer_sum += outer_sum_inc)
        for (size_t k = 0; k < indices_shape[axis]; k++)
            for (size_t inner_sum = 0; inner_sum < max_inner_sum; inner_sum++) {
                if (indices[i] < 0 || static_cast<size_t>(indices[i]) >= data_shape[axis]) {
                    throw std::domain_error{"indices values of GatherElement exceed data size"};
                }
                out[i] = data[outer_sum + max_inner_sum * indices[i] + inner_sum];
                i++;
            }
}
}  // namespace reference
}  // namespace ov
