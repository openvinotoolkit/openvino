// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/op/col2im.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v15 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Col2Im* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_shapes.size() == 3,
                           "Number of inputs has to be equal to 3. Got: ",
                           input_shapes.size());

    const auto& data_shape = input_shapes[0];
    const auto& output_size_shape = input_shapes[1];
    const auto& kernel_shape = input_shapes[2];
    const bool is_batched = data_shape.rank() == 3 || data_shape.rank().is_dynamic();
    const auto output_size_val = ov::op::get_input_const_data_as_shape<TRShape>(op, 1, tensor_accessor);
    const auto kernel_val = ov::op::get_input_const_data_as_shape<TRShape>(op, 2, tensor_accessor);

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           data_shape.rank().compatible(2) || data_shape.rank().compatible(3),
                           "input data must be an unbatched 2D or a batched 3D input. Got: ",
                           data_shape);

    if (output_size_shape.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               output_size_shape.rank().compatible(1) && output_size_shape[0].compatible(2),
                               "output_size must be a 1D input of shape [2]. Got: ",
                               output_size_shape);
    }

    if (kernel_shape.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               kernel_shape.rank().compatible(1) && kernel_shape[0].compatible(2),
                               "kernel_size must be a 1D input of shape [2]. Got: ",
                               kernel_shape);
    }

    Dimension N, C, H, W;
    if (data_shape.rank().is_dynamic()) {
        N = Dimension::dynamic();
    } else if (data_shape[0].is_dynamic()) {
        N = Dimension::dynamic();
    } else if (is_batched) {
        N = data_shape[0];
    }

    const size_t C_idx = is_batched ? 1 : 0;
    if (kernel_val && data_shape.rank().is_static() && data_shape[C_idx].is_static()) {
        const auto dividend = data_shape[C_idx].get_length();
        const auto divisor = ((*kernel_val)[0] * (*kernel_val)[1]).get_length();
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               dividend % divisor == 0,
                               "First non-batch dimension is not evenly divisible by Product(kernel_shape). Got: ",
                               data_shape[C_idx].get_length());
        C = dividend / divisor;
    } else {
        C = Dimension::dynamic();
    }

    const size_t L_idx = is_batched ? 2 : 1;
    if (data_shape.rank().is_dynamic()) {
        return {PartialShape::dynamic()};
    } else if (output_size_val && kernel_val && data_shape.rank().is_static() && data_shape[L_idx].is_static()) {
        const auto L = data_shape[L_idx].get_length();
        constexpr size_t spatial_dims = 2;

        const auto& pads_begin = op->get_pads_begin();
        const auto& pads_end = op->get_pads_end();
        const auto& strides = op->get_strides();
        const auto& dilations = op->get_dilations();

        double L_calculated = 1.0;
        for (size_t d = 0; d < spatial_dims; ++d) {
            L_calculated *= std::floor((((*output_size_val)[d].get_length() + pads_begin[d] + pads_end[d] -
                                         dilations[d] * ((*kernel_val)[d].get_length() - 1) - 1) /
                                        strides[d]) +
                                       1);
        }

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               L == L_calculated,
                               "For given inputs and parameters the total number of data blocks must be equal to " +
                                   std::to_string(static_cast<size_t>(L_calculated)) + ". Got: ",
                               L);
        H = (*output_size_val)[0];
        W = (*output_size_val)[1];
    } else {
        H = Dimension::dynamic();
        W = Dimension::dynamic();
    }

    const auto output_shape = is_batched ? TRShape{N, C, H, W} : TRShape{C, H, W};
    return {output_shape};
}
}  // namespace v15
}  // namespace op
}  // namespace ov
