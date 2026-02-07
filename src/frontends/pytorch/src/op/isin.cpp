// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_isin(const NodeContext& context) {
    // torch.isin(elements, test_elements, assume_unique=False, invert=False)
    num_inputs_check(context, 2, 4);

    auto elements = context.get_input(0);
    auto test_elements = context.get_input(1);

    bool invert = false;
    if (context.get_input_size() >= 4) {
        invert = context.const_input<bool>(3);
    }

    // -------- Core logic will go here --------

    // 1. Make sure test_elements is 1D
    auto test_shape = test_elements.get_partial_shape();
    if (test_shape.rank().is_static() && test_shape.rank().get_length() != 1) {
        test_elements = context.mark_node(
            std::make_shared<v1::Reshape>(test_elements, v0::Constant::create(element::i64, Shape{1}, {-1}), false));
    }

    // 2. Compare elements with test_elements (broadcasted)
    auto equal = context.mark_node(std::make_shared<v1::Equal>(elements, test_elements));

    // 3. Reduce with logical OR over last axis
    auto axis = v0::Constant::create(element::i64, Shape{1}, {-1});
    auto result = context.mark_node(std::make_shared<v1::ReduceLogicalOr>(equal, axis, false));

    // 4. Handle invert flag
    if (invert) {
        result = context.mark_node(std::make_shared<v1::LogicalNot>(result));
    }

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
