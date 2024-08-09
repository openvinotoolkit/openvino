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
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3);
    const auto& data_shape = input_shapes[0];
    const auto& output_size_shape = input_shapes[1];
    const auto& kernel_shape = input_shapes[2];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           ov::util::is_rank_compatible_any_of(data_shape.rank(), {2, 3}),
                           "input data must be an unbatched 2D or a batched 3D input. Got: ",
                           data_shape);

    const auto is_two_elem_1d = [](const TShape& shape) -> bool {
        static const auto exp_shape = Shape{2};
        return shape.rank().is_dynamic() || shape.to_shape() == exp_shape;
    };

    if (output_size_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               is_two_elem_1d(output_size_shape),
                               "output_size must be a 1D input of shape [2]. Got: ",
                               output_size_shape);
    }

    if (kernel_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               is_two_elem_1d(kernel_shape),
                               "kernel_size must be a 1D input of shape [2].");
    }

    if (data_shape.rank().is_static()) {
        auto output_shapes = std::vector<TRShape>(1);
        auto& output_shape = output_shapes[0];
        const bool is_batched = data_shape.rank() == 3;
        output_shape.resize(is_batched ? 4 : 3);
        size_t idx = 0;

        // output_shape: (N, C, H, W)
        //                ^
        if (is_batched) {
            output_shape[idx] = data_shape[0];
            idx++;
        }

        // output_shape: (N, C, H, W)
        //                   ^
        const size_t C_idx = is_batched ? 1 : 0;
        const auto kernel_val = ov::op::get_input_const_data_as<TRShape, int64_t>(op, 2, tensor_accessor);
        if (kernel_val && data_shape.rank().is_static() && data_shape[C_idx].is_static()) {
            const auto& dividend = data_shape[C_idx].get_length();
            const auto divisor = ((*kernel_val)[0] * (*kernel_val)[1]);
            output_shape[idx] = dividend / divisor;

            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   dividend % divisor == 0,
                                   "First non-batch dimension is not evenly divisible by Product(kernel_shape). Got: ",
                                   data_shape[C_idx].get_length());
        }

        // output_shape: (N, C, H, W)
        //                      ^  ^
        if (const auto output_size_val =
                ov::op::get_input_const_data_as_shape<TRShape, int64_t>(op, 1, tensor_accessor)) {
            idx++;
            output_shape[idx] = (*output_size_val)[0];
            idx++;
            output_shape[idx] = (*output_size_val)[1];
            const size_t L_idx = is_batched ? 2 : 1;
            if (data_shape.rank().is_static() && data_shape[L_idx].is_static() && (*output_size_val).is_static()) {
                constexpr size_t spatial_dims = 2;

                const auto& pads_begin = op->get_pads_begin();
                const auto& pads_end = op->get_pads_end();
                const auto& strides = op->get_strides();
                const auto& dilations = op->get_dilations();

                if (kernel_val) {
                    using TVal = typename TShape::value_type::value_type;
                    TVal L_calculated = 1;
                    for (size_t d = 0; d < spatial_dims; ++d) {
                        L_calculated *= (((*output_size_val)[d].get_length() + pads_begin[d] + pads_end[d] -
                                          dilations[d] * ((*kernel_val)[d] - 1) - 1) /
                                         strides[d]) +
                                        1;
                    }
                    const auto L = data_shape[L_idx].get_length();
                    NODE_SHAPE_INFER_CHECK(
                        op,
                        input_shapes,
                        L == L_calculated,
                        "For given inputs and parameters the total number of data blocks must be equal to " +
                            std::to_string(L_calculated) + ". Got: ",
                        L);
                }
            }
        }
        return output_shapes;
    } else {
        return {PartialShape::dynamic()};
    }
}
}  // namespace v15
}  // namespace op
}  // namespace ov
