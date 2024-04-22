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

    const auto output_size_val = ov::op::get_input_const_data_as_shape<TRShape>(op, 1, tensor_accessor);
    const auto kernel_val = ov::op::get_input_const_data_as_shape<TRShape>(op, 2, tensor_accessor);

    const auto& pads_begin = op->get_pads_begin();
    const auto& pads_end = op->get_pads_end();
    const auto& strides = op->get_strides();
    const auto& dilations = op->get_dilations();

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           data_shape.rank() == 2 || data_shape.rank() == 3,
                           "data_shape must be an unbatched 2D or a batched 3D input. Got: ",
                           data_shape);

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           output_size_shape.rank() == 2,
                           "output_size must be a 2D input. Got: ",
                           output_size_shape);
                    
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           kernel_shape.rank() == 2,
                           "kernel_size must be a 2D input. Got: ",
                           kernel_shape);

    const bool is_batched = data_shape.rank() == 3;
    const size_t C_idx = is_batched ? 1 : 0;
    const size_t L_idx = is_batched ? 2 : 1;

    const auto L = data_shape[L_idx].get_length();
    double L_calculated = 1.0;
    const size_t spatial_dims = 2;
    double top, bottom, wtf;
    for(size_t d = 0; d < spatial_dims; ++d) {
        const auto one = output_size_val->get_shape()[d];
        wtf = (*output_size_val)[d].get_length();
        top = wtf + pads_begin[d] + pads_end[d];
        bottom = - dilations[d] * ((*kernel_val)[d].get_length() - 1) - 1;
        L_calculated *= std::floor(((top + bottom) / strides[d]) + 1);
        //L_calculated *= std::floor(((output_size_val->at(d) + pads_begin[d] + pads_end[d] - dilations[d] * (kernel_val->at(d) - 1) - 1) / strides[d]) + 1);
    }

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           L == L_calculated,
                           "For given inputs and parameters the total number of data blocks must be equal to " + std::to_string(L_calculated) + ". Got: ",
                           L);

    size_t kernel_size_product = 1;
    for (const auto& kernel_dim: kernel_shape) {
        kernel_size_product *= kernel_dim.get_length();
    }
    
    const auto C = data_shape[C_idx] / kernel_size_product;
    const auto output_shape = is_batched
                              ? TRShape{data_shape[0], C, output_size_shape[0], output_size_shape[1]}
                              : TRShape{C, output_size_shape[0], output_size_shape[1]};

    return {output_shape};
}
}  // namespace v15
}  // namespace op
}  // namespace ov
