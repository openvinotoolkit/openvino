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
    const auto& data_shape = input_shapes[0];
    const auto& output_size_shape = input_shapes[1];
    const auto& kernel_shape = input_shapes[2];
    const bool is_batched = data_shape.rank() == 3;
    const auto output_size_val = ov::op::get_input_const_data_as_shape<TRShape>(op, 1, tensor_accessor);
    const auto kernel_val = ov::op::get_input_const_data_as_shape<TRShape>(op, 2, tensor_accessor);

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           data_shape.rank().compatible(2) || data_shape.rank().compatible(3),
                           "data_shape must be an unbatched 2D or a batched 3D input. Got: ",
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

    Dimension C;
    if (kernel_val) {
        const size_t C_idx = is_batched ? 1 : 0;
        C = data_shape[C_idx].get_length() / ((*kernel_val)[0] * (*kernel_val)[1]).get_length();
    } else {
        C = Dimension::dynamic();
    } 
    
    // Validate L only if actual values are given
    TRShape output_shape;
    if (output_size_val && kernel_val) {
        const size_t L_idx = is_batched ? 2 : 1;
        const auto L = data_shape[L_idx].get_length();
        const size_t spatial_dims = 2;

        const auto& pads_begin = op->get_pads_begin();
        const auto& pads_end = op->get_pads_end();
        const auto& strides = op->get_strides();
        const auto& dilations = op->get_dilations();

        double L_calculated = 1.0;
        for(size_t d = 0; d < spatial_dims; ++d) {
            std::cout << "output_size_val[d]: " << (*output_size_val)[d].to_string() << std::endl;
            std::cout << "kernel_val[d]: " << (*kernel_val)[d].to_string() << std::endl;
            std::cout << "pads_begin[d]: " << pads_begin[d] << std::endl;
            std::cout << "pads_end[d]: " << pads_end[d] << std::endl;
            std::cout << "dilations[d]: " << dilations[d] << std::endl;
            std::cout << "strides[d]: " << strides[d] << std::endl;
            L_calculated *= std::floor((((*output_size_val)[d].get_length() + pads_begin[d] + pads_end[d] - dilations[d] * ((*kernel_val)[d].get_length() - 1) - 1) / strides[d]) + 1);
        }

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               L == L_calculated,
                               "For given inputs and parameters the total number of data blocks must be equal to " + std::to_string(L_calculated) + ". Got: ",
                               L);
        output_shape = is_batched
                       ? TRShape{data_shape[0], C, (*output_size_val)[0], (*output_size_val)[1]}
                       : TRShape{C, (*output_size_val)[0], (*output_size_val)[1]};
    } else {
        output_shape = is_batched
                       ? TRShape{data_shape[0], C, Dimension::dynamic(), Dimension::dynamic()}
                       : TRShape{C, Dimension::dynamic(), Dimension::dynamic()};
    }

    return {output_shape};
}
}  // namespace v15
}  // namespace op
}  // namespace ov
