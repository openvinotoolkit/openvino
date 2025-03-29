// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/op/depth_to_space.hpp"
#include "openvino/opsets/opset1.hpp"
#include "utils.hpp"
namespace ov {
namespace op {
namespace v0 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const DepthToSpace* op, const std::vector<TShape>& input_shapes) {
    using TDim = typename std::iterator_traits<typename TShape::iterator>::value_type;
    using TVal = typename TShape::value_type::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);

    const auto& data_shape = input_shapes[0];
    auto output_shapes = std::vector<TRShape>{data_shape};

    if (data_shape.rank().is_static()) {
        static constexpr size_t spatial_dim_offset = 2;
        NODE_VALIDATION_CHECK(op,
                              data_shape.size() > spatial_dim_offset,
                              "The input tensor with rank lower than 3 is not supported (input rank: ",
                              data_shape.size(),
                              ")");

        const auto& block_size = op->get_block_size();
        const auto divisor = static_cast<TVal>(std::pow(block_size, data_shape.size() - spatial_dim_offset));
        NODE_VALIDATION_CHECK(op, divisor != 0, "DepthToSpace: The divisor must not be 0");

        auto& out_shape = output_shapes[0];
        out_shape[1] /= divisor;
        check_divided_result(op, out_shape[1], data_shape[1], divisor);
        std::for_each(out_shape.begin() + spatial_dim_offset, out_shape.end(), [&block_size](TDim& d) {
            d *= static_cast<TVal>(block_size);
        });
    }

    return output_shapes;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
