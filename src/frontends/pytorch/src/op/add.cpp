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
using namespace std;

namespace {
OutputVector translate_add_common(const NodeContext& context, bool inplace) {
    num_inputs_check(context, 2, 3, true);
    Output<Node> lhs = context.get_input(0);
    Output<Node> rhs = context.get_input(1);
    auto dtype0 = context.get_input_type(0);
    auto dtype1 = context.get_input_type(1);

    if (dtype0.is<type::List>() && dtype1.is<type::List>()) {
        // aten::add.t(t[] a, t[] b) -> t[]
        // Case when two lists gets concatenated
        PYTORCH_OP_CONVERSION_CHECK(false, "aten::add is used for concatenation of lists, not possible to convert");
    }
    if (inplace) {
        if (lhs.get_element_type().is_dynamic() || lhs.get_element_type() != rhs.get_element_type())
            rhs = ComplexTypeMark::convert_like(context, rhs, lhs);
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
    if (!context.input_is_none(2)) {
        alpha = context.get_input(2);
    } else if (context.has_attribute("alpha")) {
        alpha = context.get_attribute<Output<Node>>("alpha");
    }

    if (alpha.get_node_shared_ptr()) {
        auto converted_alpha = ComplexTypeMark::convert_like(context, alpha, rhs);
        rhs = ComplexTypeMark::mul(context, converted_alpha, rhs);
    }

    auto add = ComplexTypeMark::add(context, lhs, rhs);

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
