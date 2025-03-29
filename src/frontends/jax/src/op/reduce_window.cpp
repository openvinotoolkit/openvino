// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

std::tuple<std::shared_ptr<v0::Constant>,
           std::shared_ptr<v0::Constant>,
           Output<Node>,
           Strides,
           Strides,
           Shape,
           Shape,
           Shape>
reduce_window_preprocess(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    Output<Node> input = context.get_input(0);

    auto window_dimensions = context.const_named_param<std::vector<int64_t>>("window_dimensions");
    auto window_strides = context.const_named_param<Strides>("window_strides");
    auto padding = context.const_named_param<std::vector<std::vector<int64_t>>>("padding");
    auto base_dilation = context.const_named_param<Strides>("base_dilation");
    auto window_dilation = context.const_named_param<Strides>("window_dilation");
    size_t total_dim = window_dimensions.size();

    JAX_OP_CONVERSION_CHECK(window_strides.size() == total_dim,
                            "Internal error: window_strides must have the same size as window_dimensions, but got ",
                            window_strides.size(),
                            " and ",
                            total_dim);
    JAX_OP_CONVERSION_CHECK(padding.size() == total_dim,
                            "Internal error: padding must have the same size as window_dimensions, but got ",
                            padding.size(),
                            " and ",
                            total_dim);
    JAX_OP_CONVERSION_CHECK(base_dilation.size() == total_dim,
                            "Internal error: base_dilation must have the same size as window_dimensions, but got ",
                            base_dilation.size(),
                            " and ",
                            total_dim);
    JAX_OP_CONVERSION_CHECK(window_dilation.size() == total_dim,
                            "Internal error: window_dilation must have the same size as window_dimensions, but got ",
                            window_dilation.size(),
                            " and ",
                            total_dim);

    Strides strides(total_dim - 2);
    Strides dilations(total_dim - 2);
    Shape pads_begin(total_dim - 2);
    Shape pads_end(total_dim - 2);
    Shape kernel(total_dim - 2);
    for (size_t ind = 0; ind < total_dim; ++ind) {
        if (ind != 0 && ind != total_dim - 1) {
            kernel[ind - 1] = static_cast<size_t>(window_dimensions[ind]);
            pads_begin[ind - 1] = padding[ind][0];
            pads_end[ind - 1] = padding[ind][1];
            strides[ind - 1] = static_cast<size_t>(window_strides[ind]);
            dilations[ind - 1] = static_cast<size_t>(base_dilation[ind]);
        } else {
            // only support NHWC format input now.
            JAX_OP_CONVERSION_CHECK(window_dimensions[ind] == 1, "Internal error: unsupported layout.");
            JAX_OP_CONVERSION_CHECK(window_strides[ind] == 1, "Internal error: unsupported layout.");
            JAX_OP_CONVERSION_CHECK(padding[ind][0] == 0 && padding[ind][1] == 0,
                                    "Internal error: unsupported layout.");
            JAX_OP_CONVERSION_CHECK(base_dilation[ind] == 1, "Internal error: unsupported layout.");
        }
        JAX_OP_CONVERSION_CHECK(window_dilation[ind] == 1, "Internal error: only window_dilation 1 is supported.");
    }

    std::vector<int64_t> in_transpose_vector(total_dim);
    std::vector<int64_t> out_transpose_vector(total_dim);
    in_transpose_vector[0] = 0;
    in_transpose_vector[1] = total_dim - 1;
    out_transpose_vector[0] = 0;
    out_transpose_vector[total_dim - 1] = 1;
    for (size_t i = 2; i < total_dim; i++) {
        in_transpose_vector[i] = i - 1;
        out_transpose_vector[i - 1] = i;
    }
    auto input_transpose_order =
        std::make_shared<v0::Constant>(element::i64, Shape{in_transpose_vector.size()}, in_transpose_vector);
    auto output_transpose_order =
        std::make_shared<v0::Constant>(element::i64, Shape{out_transpose_vector.size()}, out_transpose_vector);
    return {input_transpose_order, output_transpose_order, input, strides, dilations, pads_begin, pads_end, kernel};
};

OutputVector translate_reduce_window_max(const NodeContext& context) {
    auto elements = reduce_window_preprocess(context);
    auto input_transpose_order = std::get<0>(elements);
    auto output_transpose_order = std::get<1>(elements);
    auto input = std::get<2>(elements);
    auto strides = std::get<3>(elements);
    auto dilations = std::get<4>(elements);
    auto pads_begin = std::get<5>(elements);
    auto pads_end = std::get<6>(elements);
    auto kernel = std::get<7>(elements);

    input = std::make_shared<v1::Transpose>(input, input_transpose_order);
    Output<Node> res = std::make_shared<v14::MaxPool>(input, strides, dilations, pads_begin, pads_end, kernel);
    res = std::make_shared<v1::Transpose>(res, output_transpose_order);
    return {res};
}

OutputVector translate_reduce_window_sum(const NodeContext& context) {
    auto elements = reduce_window_preprocess(context);
    auto input_transpose_order = std::get<0>(elements);
    auto output_transpose_order = std::get<1>(elements);
    auto input = std::get<2>(elements);
    auto strides = std::get<3>(elements);
    auto dilations = std::get<4>(elements);
    auto pads_begin = std::get<5>(elements);
    auto pads_end = std::get<6>(elements);
    auto kernel = std::get<7>(elements);

    input = std::make_shared<v1::Transpose>(input, input_transpose_order);
    Output<Node> res = std::make_shared<v14::AvgPool>(input, strides, pads_begin, pads_end, kernel, false);
    res = std::make_shared<v1::Transpose>(res, output_transpose_order);
    auto kernel_size = ov::shape_size(kernel);
    Output<Node> kernel_size_constant = std::make_shared<v0::Constant>(res.get_element_type(), Shape{}, kernel_size);
    res = std::make_shared<v1::Multiply>(res, kernel_size_constant);
    return {res};
}

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov