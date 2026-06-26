// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "dimension_util.hpp"
#include "openvino/op/bincount.hpp"
#include "openvino/reference/bincount.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v17 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Bincount* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 || input_shapes.size() == 2);

    const auto& data_shape = input_shapes[0];
    NODE_VALIDATION_CHECK(op,
                          data_shape.rank().compatible(1),
                          "The 'data' input must be a 1-D tensor. Got rank: ",
                          data_shape.rank());

    if (input_shapes.size() == 2) {
        const auto& weights_shape = input_shapes[1];
        NODE_VALIDATION_CHECK(op,
                              weights_shape.rank().compatible(1),
                              "The 'weights' input must be a 1-D tensor. Got rank: ",
                              weights_shape.rank());
        if (data_shape.rank().is_static() && weights_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  data_shape[0].compatible(weights_shape[0]),
                                  "The 'data' and 'weights' inputs must have the same length. Got: ",
                                  data_shape[0],
                                  " vs ",
                                  weights_shape[0]);
        }
    }

    if (const auto data_tensor = ta(0)) {
        const auto data_et = data_tensor.get_element_type();
        const auto n = data_tensor.get_size();
        size_t out_size = 0;

        switch (data_et) {
        case element::i32:
            out_size =
                reference::bincount_output_size(data_tensor.data<const int32_t>(), n, op->get_minlength());
            break;
        case element::i64:
            out_size =
                reference::bincount_output_size(data_tensor.data<const int64_t>(), n, op->get_minlength());
            break;
        case element::u8:
            out_size =
                reference::bincount_output_size(data_tensor.data<const uint8_t>(), n, op->get_minlength());
            break;
        case element::u16:
            out_size =
                reference::bincount_output_size(data_tensor.data<const uint16_t>(), n, op->get_minlength());
            break;
        case element::u32:
            out_size =
                reference::bincount_output_size(data_tensor.data<const uint32_t>(), n, op->get_minlength());
            break;
        case element::u64:
            out_size =
                reference::bincount_output_size(data_tensor.data<const uint64_t>(), n, op->get_minlength());
            break;
        default:
            break;
        }

        // We have constant data — return exact output size
        return {TRShape{static_cast<typename TRShape::value_type>(out_size)}};
    }

    // Dynamic case: use minlength as lower bound
    using TDim = typename TRShape::value_type;
    return {TRShape{TDim(op->get_minlength(), ov::util::dim::inf_bound)}};
}

}  // namespace v17
}  // namespace op
}  // namespace ov
