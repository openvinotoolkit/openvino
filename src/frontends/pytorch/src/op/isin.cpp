// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "utils.hpp"
namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_isin(const NodeContext& context) {
    // aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False)
    // ->Tensor
    // aten::isin.Tensor_Scalar(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool
    // invert=False) -> Tensor
    // aten::isin.Scalar_Tensor(Scalar element, Tensor test_elements, *, bool assume_unique=False,
    // bool invert=False) -> Tensor
    num_inputs_check(context, 2, 4);
    auto elements = context.get_input(0);
    auto test_element = context.get_input(1);
    bool invert = !context.input_is_none(3) ? context.const_input<bool>(3) : false;
    auto neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto shape0 = context.mark_node(std::make_shared<v3::ShapeOf>(elements, element::i32));
    auto shape1 = context.mark_node(v0::Constant::create(element::i32, Shape{shape0->get_output_size()}, {1}));
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{shape1, neg_1}, 0);
    std::shared_ptr<ov::Node> result;
    if (!invert) {
        result = context.mark_node(std::make_shared<v1::ReduceLogicalOr>(
            std::make_shared<v1::Equal>(std::make_shared<v0::Unsqueeze>(elements, neg_1),
                                        std::make_shared<v1::Reshape>(test_element, shape2, false)),
            neg_1,
            false));
    } else {
        result = context.mark_node(std::make_shared<v1::ReduceLogicalAnd>(
            std::make_shared<v1::NotEqual>(std::make_shared<v0::Unsqueeze>(elements, neg_1),
                                           std::make_shared<v1::Reshape>(test_element, shape2, false)),
            neg_1,
            false));
    }
    return {result};
}
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
