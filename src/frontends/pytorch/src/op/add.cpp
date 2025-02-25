// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
using namespace ov::frontend;
using namespace std;

namespace {
OutputVector translate_add_common(const NodeContext& context, bool inplace) {
    num_inputs_check(context, 2, 3, true);
    Output<Node> lhs = context.get_input(0);
    Output<Node> rhs = context.get_input(1);
    auto dtype0 = context.get_input_type(0);
    auto dtype1 = context.get_input_type(1);

    auto lhs_complex = ov::as_type_ptr<ComplexTypeMark>(lhs.get_node_shared_ptr());
    auto is_lhs_complex = (lhs_complex != nullptr);
    auto rhs_complex = ov::as_type_ptr<ComplexTypeMark>(rhs.get_node_shared_ptr());
    auto is_rhs_complex = (rhs_complex != nullptr);

    if (lhs_complex) {
        lhs = lhs_complex->input_value(0);
    }
    if (rhs_complex) {
        rhs = rhs_complex->input_value(0);
    }

    if (dtype0.is<type::List>() && dtype1.is<type::List>()) {
        // aten::add.t(t[] a, t[] b) -> t[]
        // Case when two lists gets concatenated
        PYTORCH_OP_CONVERSION_CHECK(false, "aten::add is used for concatenation of lists, not possible to convert");
    }
    if (inplace) {
        if (lhs.get_element_type().is_dynamic() || lhs.get_element_type() != rhs.get_element_type())
            rhs = context.mark_node(std::make_shared<v1::ConvertLike>(rhs, lhs));
    } else {
        align_eltwise_input_types(context,
                                  lhs,
                                  rhs,
                                  is_python_scalar_input(context, 0),
                                  is_python_scalar_input(context, 1));
    }

    auto left_is_bool = lhs.get_element_type() == ov::element::boolean ||
                        (dtype0.is<element::Type>() && dtype0.as<element::Type>() == element::boolean);
    auto right_is_bool = rhs.get_element_type() == ov::element::boolean ||
                         (dtype1.is<element::Type>() && dtype1.as<element::Type>() == element::boolean);

    if (left_is_bool && right_is_bool) {
        // when types are boolean then add means logical Or operation
        auto logical_or = context.mark_node(std::make_shared<v1::LogicalOr>(lhs, rhs));
        if (inplace)
            context.mutate_input(0, logical_or);

        return {logical_or};
    }

    Output<Node> alpha;
    shared_ptr<ComplexTypeMark> alpha_complex = nullptr;
    bool is_alpha_complex = false;
    if (!context.input_is_none(2)) {
        alpha = context.get_input(2);
        alpha_complex = ov::as_type_ptr<ComplexTypeMark>(alpha.get_node_shared_ptr());
    } else if (context.has_attribute("alpha")) {
        alpha = context.get_attribute<Output<Node>>("alpha");
        alpha_complex = ov::as_type_ptr<ComplexTypeMark>(alpha.get_node_shared_ptr());
    }

    if (alpha_complex) {
        alpha = alpha_complex->input_value(0);
        is_alpha_complex = true;
    }

    if (alpha.get_node_shared_ptr()) {
        auto converted_alpha = context.mark_node(std::make_shared<v1::ConvertLike>(alpha, rhs));
        rhs = ComplexTypeMark::mul(context, rhs, converted_alpha, is_rhs_complex, is_alpha_complex);
        is_rhs_complex |= is_alpha_complex;
    }

    auto add = ComplexTypeMark::add(context, lhs, rhs, is_lhs_complex, is_rhs_complex);

    if (is_lhs_complex || is_rhs_complex) {
        add = context.mark_node(make_shared<ComplexTypeMark>(add, add.get_element_type()));
    }

    if (inplace)
        context.mutate_input(0, add);

    return {add};
};
}  // namespace

OutputVector translate_add(const NodeContext& context) {
    return translate_add_common(context, false);
};

OutputVector translate_add_(const NodeContext& context) {
    return translate_add_common(context, true);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
