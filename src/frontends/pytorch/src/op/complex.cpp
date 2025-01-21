// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_complex(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto real = context.get_input(0);
    auto imag = context.get_input(1);

    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    real = context.mark_node(std::make_shared<v0::Unsqueeze>(real, const_neg_1));
    imag = context.mark_node(std::make_shared<v0::Unsqueeze>(imag, const_neg_1));

    auto complex = context.mark_node(std::make_shared<v0::Concat>(OutputVector{real, imag}, -1));

    return {context.mark_node(std::make_shared<ComplexTypeMark>(complex, complex->get_element_type()))};
};

OutputVector translate_imag(const NodeContext& context) {
    num_inputs_check(context, 1, 1, true);
    auto complex = context.get_input(0);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(complex.get_node_shared_ptr());
    PYTORCH_OP_CONVERSION_CHECK(complex_type_mark, "aten::imag operation expects complex type tensor on input.");

    complex = complex_type_mark->input_value(0);
    auto axis = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto imag = context.mark_node(std::make_shared<v1::Split>(complex, axis, 2))->output(1);

    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    return {context.mark_node(std::make_shared<v0::Squeeze>(imag, const_neg_1))};
};

OutputVector translate_real(const NodeContext& context) {
    num_inputs_check(context, 1, 1, true);
    auto complex = context.get_input(0);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(complex.get_node_shared_ptr());
    PYTORCH_OP_CONVERSION_CHECK(complex_type_mark, "aten::real operation expects complex type tensor on input.");

    complex = complex_type_mark->input_value(0);
    auto axis = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto real = context.mark_node(std::make_shared<v1::Split>(complex, axis, 2))->output(0);

    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    return {context.mark_node(std::make_shared<v0::Squeeze>(real, const_neg_1))};
};

OutputVector translate_view_as_real(const NodeContext& context) {
    num_inputs_check(context, 1, 1, true);
    auto complex = context.get_input(0);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(complex.get_node_shared_ptr());
    PYTORCH_OP_CONVERSION_CHECK(complex_type_mark, "aten::real operation expects complex type tensor on input.");

    return {complex_type_mark->input_value(0)};
};

OutputVector translate_view_as_complex(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto complex = context.get_input(0);

    return {context.mark_node(std::make_shared<ComplexTypeMark>(complex, complex.get_element_type()))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
