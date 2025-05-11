// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector trilu(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    const auto num_inputs = inputs.size();

    CHECK_VALID_NODE(node, num_inputs > 0 && num_inputs <= 2, "Trilu expects <= 2 input tensors. Got: ", num_inputs);

    const auto& input = inputs[0];
    const auto& rank = input.get_partial_shape().rank();
    if (rank.is_static()) {
        CHECK_VALID_NODE(node, rank.get_length() >= 2, "Trilu first input's rank must be >= 2");
    }

    Output<ov::Node> k;
    bool is_k_available = num_inputs == 2 && !ov::op::util::is_null(inputs[1]);
    if (is_k_available) {
        // Trilu-14 documentation allows only 0-D tensor (scalar),
        // but we extend support to be able work with 1-D with length == 1
        k = inputs[1];
        auto axes = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        // Check if k is a tensor with a single value
        if (k.get_shape().size() == 1 && k.get_shape()[0] == 1) {
            k = std::make_shared<v0::Squeeze>(k, axes);
        }
        CHECK_VALID_NODE(node, k.get_partial_shape().compatible({}), "Trilu second input must be a scalar");
    }

    const auto shape = std::make_shared<v3::ShapeOf>(input);
    const auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    const auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

    // The approach here is to create a mask, that later can be used in Select operator
    // to choose appropriate values from the input
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
    const auto M = std::make_shared<v8::Gather>(shape, v0::Constant::create(ov::element::i32, ov::Shape{}, {-1}), zero);
    const auto N = std::make_shared<v8::Gather>(shape, v0::Constant::create(ov::element::i32, ov::Shape{}, {-2}), zero);

    // create 2D tensor with shape [1, M] and values [[0, 1, ..., M - 1]]
    const auto horizontal_range =
        std::make_shared<v0::Unsqueeze>(std::make_shared<v4::Range>(zero, M, one, ov::element::i64), zero);
    // create 2D tensor with shape [N, 1] and values [[k], [k + 1], ..., [N + k - 1]]
    std::shared_ptr<ov::Node> vertical_range;
    if (is_k_available) {
        vertical_range = std::make_shared<v4::Range>(k, std::make_shared<v1::Add>(N, k), one, ov::element::i64);
    } else {
        vertical_range = std::make_shared<v4::Range>(zero, N, one, ov::element::i64);
    }
    vertical_range = std::make_shared<v0::Unsqueeze>(vertical_range, one);

    const bool upper = node.get_attribute_value<int64_t>("upper", 1) == 1;
    std::shared_ptr<ov::Node> mask;
    if (upper) {
        mask = std::make_shared<v1::GreaterEqual>(horizontal_range, vertical_range);
    } else {
        mask = std::make_shared<v1::LessEqual>(horizontal_range, vertical_range);
    }

    return {
        std::make_shared<v1::Select>(mask, input, v0::Constant::create(input.get_element_type(), ov::Shape{}, {0}))};
}

ONNX_OP("Trilu", OPSET_SINCE(1), ai_onnx::opset_1::trilu);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
