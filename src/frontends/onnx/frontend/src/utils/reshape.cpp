// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/builder/reshape.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>

#include "default_opset.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/shape.hpp"
#include "utils/reshape.hpp"

namespace ngraph {
namespace onnx_import {
namespace reshape {
std::vector<std::size_t> infer_dimensions(const std::string& node_name,
                                          const std::vector<std::size_t>& input_shape,
                                          const std::vector<std::size_t>& output_shape) {
    std::vector<std::size_t> inferred_dims{output_shape};

    // If an output dimension is equal to zero its actual value is copied from the input
    // shape argument.
    for (std::size_t idx = 0; idx < inferred_dims.size(); ++idx) {
        if (inferred_dims.at(idx) == 0) {
            NGRAPH_CHECK(idx < input_shape.size(),
                         "Node ",
                         node_name,
                         " cannot copy dimension from the input data shape because "
                         "requested index is out of range.");

            inferred_dims.at(idx) = input_shape.at(idx);
        }
    }

    // Check whether there are dimensions equal to -1 in output_shape. There may be at
    // most one such case. Its value is then inferred from the size of the tensor and
    // the remaining dimensions.
    auto neg_value_it = std::find(std::begin(inferred_dims), std::end(inferred_dims), -1);
    if (neg_value_it != std::end(inferred_dims)) {
        // only single '-1' value is allowed
        NGRAPH_CHECK(std::find(std::next(neg_value_it), std::end(inferred_dims), -1) == std::end(inferred_dims),
                     "Node ",
                     node_name,
                     " more than one dimension is set to (-1). ",
                     "Only one dimension value can be inferred.");

        // Set dimension value to 1 temporarily to be able to calculate its value.
        *neg_value_it = 1;
        std::size_t input_shape_product =
            std::accumulate(std::begin(input_shape), std::end(input_shape), size_t{1}, std::multiplies<std::size_t>());
        std::size_t output_shape_product = std::accumulate(std::begin(inferred_dims),
                                                           std::end(inferred_dims),
                                                           size_t{1},
                                                           std::multiplies<std::size_t>());
        *neg_value_it = input_shape_product / output_shape_product;
    }

    return inferred_dims;
}

Output<ngraph::Node> interpret_as_scalar(const Output<ngraph::Node>& node) {
    Shape node_shape = node.get_partial_shape().to_shape();

    // If node is already a scalar, return original
    if (is_scalar(node_shape)) {
        return node;
    }

    NGRAPH_CHECK((shape_size(node_shape) == 1), "Scalar value can't be derived from a node with ", node_shape);

    // If node is a Constant, recreate as Constant with Shape{}
    if (ngraph::op::is_constant(node.get_node())) {
        const auto value = ngraph::as_type_ptr<default_opset::Constant>(node.get_node_shared_ptr())->get_data_ptr();
        return std::make_shared<default_opset::Constant>(node.get_element_type(), ngraph::Shape{}, value);
    }

    return builder::opset1::reshape(node, Shape{});
}

Output<ngraph::Node> reshape_channel_shaped_node_to_nchw(const Output<ngraph::Node>& node,
                                                         const Output<ngraph::Node>& expected_rank) {
    // Prepare tail shape (rank = conv.rank - 2): [1, 1, 1, 1, ... ]
    const auto one_const = default_opset::Constant::create(element::i64, Shape{1}, {1});
    const auto two_const = default_opset::Constant::create(element::i64, Shape{1}, {2});
    const auto tail_shape_rank = std::make_shared<default_opset::Subtract>(expected_rank, two_const);
    const auto tail_shape = std::make_shared<default_opset::Broadcast>(one_const, tail_shape_rank);

    // Construct new bias shape: [1, C, 1, 1, ... ]
    const auto C_dim = std::make_shared<default_opset::ShapeOf>(node);
    const auto new_shape = std::make_shared<default_opset::Concat>(OutputVector{one_const, C_dim, tail_shape}, 0);

    return std::make_shared<default_opset::Reshape>(node, new_shape, false);
}

}  // namespace  reshape
}  // namespace onnx_import
}  // namespace ngraph
