// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/op/string_tensor_pack.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v15 {
namespace util {
template <class TShape, class TRShape = result_shape_t<TShape>>
static void validate_indices(const size_t input_index,
                             const ITensorAccessor& tensor_accessor,
                             const StringTensorPack* op,
                             const std::vector<TShape>& input_shapes) {
    const auto& indices_shape = input_shapes[input_index];
    if (indices_shape.is_static()) {
        const auto data = ov::op::get_input_const_data_as<TRShape, int64_t>(op, input_index, tensor_accessor);
        if (data) {
            const auto element_count = data->size();
            if (element_count > 0) {
                NODE_SHAPE_INFER_CHECK(op, input_shapes, (*data)[0] >= 0, "Indices cannot be negative.");
                const auto& symbols_shape = input_shapes[2];
                if (symbols_shape.is_static()) {
                    // ends indices are offset by 1
                    const auto offset = input_index == 0 ? 0 : 1;
                    NODE_SHAPE_INFER_CHECK(
                        op,
                        input_shapes,
                        static_cast<unsigned long int>(data->back()) <
                            static_cast<unsigned long int>(symbols_shape[0].get_length() + offset),
                        "The biggest index cannot be higher than the amount or characters in symbols input.");
                }
                if (element_count > 1) {
                    bool indices_ascending = true;
                    for (size_t i = 1; i < element_count; i++) {
                        if ((*data)[i] < (*data)[i - 1]) {
                            indices_ascending = false;
                            break;
                        }
                    }
                    NODE_SHAPE_INFER_CHECK(op, input_shapes, indices_ascending, "Indices must be in ascending order.");
                }
            }
        }
    }
}
}  // namespace util
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const StringTensorPack* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes.size() == 3);
    const auto& begins_shape = input_shapes[0];
    const auto& ends_shape = input_shapes[1];
    NODE_SHAPE_INFER_CHECK(op, input_shapes, begins_shape.compatible(ends_shape));
    const auto& symbols_shape = input_shapes[2];
    NODE_SHAPE_INFER_CHECK(op, input_shapes, symbols_shape.rank().compatible(1), "Symbols input must be 1D.");
    util::validate_indices(0, tensor_accessor, op, input_shapes);
    util::validate_indices(1, tensor_accessor, op, input_shapes);
    // if (begins_shape.is_static()) {
    //     const auto begins_data = ov::op::get_input_const_data_as<TRShape, int64_t>(op, 0, tensor_accessor);
    //     if (begins_data) {
    //         bool indices_ascending = true;
    //         const auto element_count = begins_data->size();
    //         if (element_count > 0) {
    //             NODE_SHAPE_INFER_CHECK(op,
    //                                input_shapes,
    //                                (*begins_data)[0] >= 0,
    //                                "Begins indices cannot be negative.");
    //             if (symbols_shape.is_static()) {
    //                 NODE_SHAPE_INFER_CHECK(op,
    //                                 input_shapes,
    //                                 static_cast<unsigned long int>(begins_data->back()) < static_cast<unsigned long
    //                                 int>(symbols_shape[0].get_length()), "The biggest index cannot be higher than the
    //                                 amount or characters in symbols input.");
    //                 }
    //         if (element_count > 1) {
    //             for (size_t i = 1; i < element_count; i++) {
    //                 if ((*begins_data)[i] < (*begins_data)[i - 1]) {
    //                     indices_ascending = false;
    //                     break;
    //                 }
    //             }
    //             NODE_SHAPE_INFER_CHECK(op,
    //                                 input_shapes,
    //                                 indices_ascending,
    //                                 "Begins indices must be in ascending order.");
    //             }
    //         }
    //     }
    // }
    return {begins_shape};
}
}  // namespace v15
}  // namespace op
}  // namespace ov
