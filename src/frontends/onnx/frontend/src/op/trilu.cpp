// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/trilu.hpp"

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "onnx_import/core/null_node.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector trilu(const Node& node) {
    const auto inputs = node.get_ng_inputs();
    const auto num_inputs = inputs.size();

    CHECK_VALID_NODE(node, num_inputs > 0 && num_inputs <= 2, "Trilu expects <= 2 input tensors. Got: ", num_inputs);

    const auto& input = inputs[0];
    const auto& rank = input.get_partial_shape().rank();
    if (rank.is_static()) {
        CHECK_VALID_NODE(node, rank.get_length() >= 2, "Trilu first input's rank must be >= 2");
    }
    bool is_k_available = num_inputs == 2 && !ngraph::op::is_null(inputs[1]);
    if (is_k_available) {
        CHECK_VALID_NODE(node, inputs[1].get_partial_shape().compatible({}), "Trilu second input must be a scalar");
    }

    const auto shape = std::make_shared<default_opset::ShapeOf>(input);
    const auto zero = default_opset::Constant::create(element::i64, Shape{}, {0});
    const auto one = default_opset::Constant::create(element::i64, Shape{}, {1});

    // The approach here is to create a mask, that later can be used in Select operator
    // to choose appropiate values from the input
    //
    // For example N = 4, M = 5, k = -1, upper = false
    // horizontal_range = [[0, 1, 2, 3, 4]]
    // vertical_range = [[-1],
    //                   [0],
    //                   [1],
    //                   [2]]
    // Since upper == false, we compare horizontal_range <= vertical_range
    // and thanks to broadcasting, it conceptually looks like:
    // [[0, 1, 2, 3, 4],        [[-1, -1, -1, -1, -1],
    //  [0, 1, 2, 3, 4],   <=    [0, 0, 0, 0, 0],
    //  [0, 1, 2, 3, 4],         [1, 1, 1, 1, 1],
    //  [0, 1, 2, 3, 4]]         [2, 2, 2, 2, 2]]
    //
    // which results in following mask:
    // [[0, 0, 0, 0, 0],
    //  [1, 0, 0, 0, 0],
    //  [1, 1, 0, 0, 0],
    //  [1, 1, 1, 0, 0]]
    //
    // That matches ONNX spec: "If upper is set to false, ... a negative k value excludes the main diagonal
    // and (|k|-1) diagonals below it"

    // fetch last two dimensions of input shape
    // M = shape[-1]
    // N = shape[-2]
    const auto M = std::make_shared<default_opset::Gather>(shape,
                                                           default_opset::Constant::create(element::i32, Shape{}, {-1}),
                                                           zero);
    const auto N = std::make_shared<default_opset::Gather>(shape,
                                                           default_opset::Constant::create(element::i32, Shape{}, {-2}),
                                                           zero);

    // create 2D tensor with shape [1, M] and values [[0, 1, ..., M - 1]]
    const auto horizontal_range =
        std::make_shared<default_opset::Unsqueeze>(std::make_shared<default_opset::Range>(zero, M, one, element::i64),
                                                   zero);
    // create 2D tensor with shape [N, 1] and values [[k], [k + 1], ..., [N + k - 1]]
    std::shared_ptr<ngraph::Node> vertical_range;
    if (is_k_available) {
        vertical_range = std::make_shared<default_opset::Range>(inputs[1],
                                                                std::make_shared<default_opset::Add>(N, inputs[1]),
                                                                one,
                                                                element::i64);
    } else {
        vertical_range = std::make_shared<default_opset::Range>(zero, N, one, element::i64);
    }
    vertical_range = std::make_shared<default_opset::Unsqueeze>(vertical_range, one);

    const bool upper = node.get_attribute_value<int64_t>("upper", 1) == 1;
    std::shared_ptr<ngraph::Node> mask;
    if (upper) {
        mask = std::make_shared<default_opset::GreaterEqual>(horizontal_range, vertical_range);
    } else {
        mask = std::make_shared<default_opset::LessEqual>(horizontal_range, vertical_range);
    }

    return {std::make_shared<default_opset::Select>(
        mask,
        input,
        default_opset::Constant::create(input.get_element_type(), Shape{}, {0}))};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
