// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <cmath>

#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
template <typename dataType>
void scatter_update(const dataType* input_data,
                    const int64_t* indices,
                    const dataType* updates,
                    const int64_t axis,
                    dataType* out_buf,
                    const size_t elem_size,
                    const Shape& data_shape,
                    const Shape& indices_shape,
                    const Shape& updates_shape) {
    std::memcpy(out_buf, input_data, sizeof(dataType) * shape_size(data_shape));

    const auto num_of_updates = shape_size(indices_shape);
    const auto size_before_axis = shape_size(Shape(data_shape.begin(), data_shape.begin() + axis));
    const auto size_after_axis = shape_size(Shape(data_shape.begin() + axis + 1, data_shape.end()));
    const auto axis_dim = data_shape[axis];

    // const auto updates_axis = updates_shape[axis] == 1 ? axis + 1 : axis;
    // const auto updates_axis_dim = updates_shape[updates_axis];
    // const auto updates_axis_last_dim = updates_shape.size() - updates_axis == 1;
    // const auto swaps = updates_axis_last_dim ? updates_shape[axis - 1] : updates_shape[axis + 1];
    // // const auto swaps = updates_axis_last_dim ? 1 ;

    const auto updates_size_before_axis =
        shape_size(Shape(updates_shape.begin(), updates_shape.begin() + axis));
    const auto updates_size_after_axis =
        shape_size(Shape(updates_shape.begin() + axis + 1, updates_shape.end()));

    // const auto updates_jump =  axis == 0 ? updates_shape.back() : 1;


    // std::vector<int64_t> updates_ids;
    // const auto updates_space = num_of_updates * updates_size_before_axis * updates_size_after_axis;
    // updates_ids.reserve(updates_space);
    std::vector<int64_t> data_ids;
    const auto data_space = num_of_updates * size_before_axis * size_after_axis;
    data_ids.reserve(data_space);

    // const auto moves = [&] {
    //     const auto needed = size_after_axis * num_of_updates;
    //     auto changes =  int(size_after_axis);
    //     const auto updates_num = shape_size(updates_shape);
    //     if (updates_num < needed) {
    //         auto left = std::remainder(updates_num, needed);
    //         left = left >= 0 ? left : -left;
    //         auto changes = int((needed - left - size_after_axis) / swaps);
    //         return changes;
    //     }
    //     return changes;

    // }();
    // const auto data_axis_last_dim = axis + 1 == data_shape.size();
    // const auto less = data_axis_last_dim ? false : updates_shape.back() < data_shape.back();
    // const auto data_dims = shape_size(data_shape);
    // const auto updates_dims = shape_size(updates_shape);
    // const auto less = ((updates_shape.back() < data_shape.back()) && (updates_dims >= data_dims));

    // int check = 0;
    // // const bool less = false;
    // int dim_after_axis{0};
    // int dim_before_axis{-1};
    // int update_num{-1};
    // for (size_t k = 0; k < num_of_updates; ++k) {
    //     const auto ind_idx = *(indices + k);
    //     ++update_num;
    //     dim_before_axis = less ? 0 : -1;
    //     dim_after_axis = 0;
    //     for (size_t i = 0; i < size_before_axis; ++i) {
    //         const auto slice_idx = i * axis_dim * size_after_axis;
    //         if (!less) {
    //             ++dim_before_axis;
    //             dim_after_axis = 0;
    //         }
    //         for (size_t j = 0; j < moves; ++j) {
    //             const auto sequence_start_idx = slice_idx + j;
    //             const auto up_slice_idx = dim_before_axis * updates_axis_dim * updates_size_after_axis;
    //             const auto up_sequence_start_idx = up_slice_idx + dim_after_axis;
    //             const auto element_idx = sequence_start_idx + (ind_idx * size_after_axis);
    //             const auto updates_idx = up_sequence_start_idx + update_num * updates_jump;

    //             if (updates_idx >= shape_size(updates_shape))
    //                 break;
                
    //             if (element_idx == size_after_axis) {
    //                 check = 1;
    //             }
    //             updates_ids.push_back(updates_idx);
    //             data_ids.push_back(element_idx);

    //             ++dim_after_axis;
    //             if (less) {
    //                 if (dim_after_axis == swaps) {
    //                     ++dim_before_axis;
    //                     dim_after_axis = 0;
    //                 };
    //                 if (dim_before_axis == updates_size_before_axis) {
    //                     dim_before_axis = 0;
    //                     ++update_num;
    //             }
    //             }
    //         }
    //     }
    // }
    const auto updates_axis_sum = updates_size_before_axis * updates_size_after_axis;
    const auto data_axis_sum = size_before_axis * size_after_axis;
    const auto udpates_smaller =  updates_axis_sum < data_axis_sum ? updates_axis_sum : data_axis_sum;
    const auto eq = udpates_smaller ? updates_axis_sum : data_axis_sum;
    const auto data_ndim = data_shape.size();
    const auto updates_ndim = updates_shape.size();
    const auto final_before_axis = std::min(size_before_axis, updates_size_before_axis);
    const auto final_after_axis = ((data_ndim != updates_ndim) && (axis == 0)) ? updates_shape.back() : std::min(size_after_axis, updates_size_after_axis);
    const auto final_num_of_updates = num_of_updates <= eq ? num_of_updates : eq; 
    for (size_t j = 0; j < final_before_axis; ++j) {
        for (size_t k = 0; k < final_num_of_updates; ++k) {
            const auto ind_idx = *(indices + k);
            for (size_t i = 0; i < final_after_axis; ++i) {
                const auto elem_idx = j * axis_dim * size_after_axis + ind_idx * size_after_axis + i;
                data_ids.push_back(elem_idx);
            }
        }
    }

    for (size_t i = 0; i < data_ids.size(); ++i) {
        out_buf[data_ids[i]] = 4;
    }

}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
 