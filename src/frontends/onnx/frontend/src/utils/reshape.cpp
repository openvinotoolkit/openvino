// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/ov_builders/reshape.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>

#include "openvino/frontend/exception.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/op_types.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;

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
            FRONT_END_GENERAL_CHECK(idx < input_shape.size(),
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
        FRONT_END_GENERAL_CHECK(
            std::find(std::next(neg_value_it), std::end(inferred_dims), -1) == std::end(inferred_dims),
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

Output<ov::Node> interpret_as_scalar(const Output<ov::Node>& node) {
    Shape node_shape = node.get_shape();

    // If node is already a scalar, return original
    if (is_scalar(node_shape)) {
        return node;
    }

    FRONT_END_GENERAL_CHECK((shape_size(node_shape) == 1),
                            "Scalar value can't be derived from a node with ",
                            node_shape);

    // If node is a Constant, recreate as Constant with Shape{}
    if (ov::op::util::is_constant(node.get_node())) {
        const auto value = ov::as_type_ptr<v0::Constant>(node.get_node_shared_ptr())->get_data_ptr();
        return std::make_shared<v0::Constant>(node.get_element_type(), ov::Shape{}, value);
    }

    return ov::op::util::reshape(node, Shape{});
}

Output<ov::Node> reshape_channel_shaped_node_to_nchw(const Output<ov::Node>& node,
                                                     const Output<ov::Node>& expected_rank) {
    // Prepare tail shape (rank = conv.rank - 2): [1, 1, 1, 1, ... ]
    const auto one_const = v0::Constant::create(element::i64, Shape{1}, {1});
    const auto two_const = v0::Constant::create(element::i64, Shape{1}, {2});
    const auto tail_shape_rank = std::make_shared<v1::Subtract>(expected_rank, two_const);
    const auto tail_shape = std::make_shared<v3::Broadcast>(one_const, tail_shape_rank);

    // Construct new bias shape: [1, C, 1, 1, ... ]
    const auto C_dim = std::make_shared<v3::ShapeOf>(node);
    const auto new_shape = std::make_shared<v0::Concat>(OutputVector{one_const, C_dim, tail_shape}, 0);

    return std::make_shared<v1::Reshape>(node, new_shape, false);
}

}  // namespace  reshape
}  // namespace onnx_import
}  // namespace ngraph
