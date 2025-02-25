// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov;
using namespace ov::op;
using namespace std;

OutputVector translate_mul_common(const NodeContext& context, bool inplace) {
    num_inputs_check(context, 2, 2, true);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    auto lhs_dtype = simplified_type_interpret(context.get_input_type(0));
    auto rhs_dtype = simplified_type_interpret(context.get_input_type(1));

    auto lhs_complex = ov::as_type_ptr<ComplexTypeMark>(lhs.get_node_shared_ptr());
    auto is_lhs_complex = (lhs_complex != nullptr);
    auto rhs_complex = ov::as_type_ptr<ComplexTypeMark>(rhs.get_node_shared_ptr());
    auto is_rhs_complex = (rhs_complex != nullptr);

    if (is_lhs_complex) {
        lhs = lhs_complex->input_value(0);
    }
    if (is_rhs_complex) {
        rhs = rhs_complex->input_value(0);
    }

    auto left_is_bool = lhs.get_element_type() == ov::element::boolean ||
                        (lhs_dtype.is<element::Type>() && lhs_dtype.as<element::Type>() == element::boolean);
    auto right_is_bool = rhs.get_element_type() == ov::element::boolean ||
                         (rhs_dtype.is<element::Type>() && rhs_dtype.as<element::Type>() == element::boolean);

    if (left_is_bool && right_is_bool) {
        // if input types are boolean then aten::mul means logical Add operation
        if (inplace)
            return op::inplace_translate_1to1_match_2_inputs_align_types<v1::LogicalAnd>(context);
        else
            return op::translate_1to1_match_2_inputs_align_types<v1::LogicalAnd>(context);
    }

    if (inplace) {
        // For inplace op we know direction of type alignment
        if (lhs.get_element_type().is_dynamic() || lhs.get_element_type() != rhs.get_element_type())
            rhs = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(rhs, lhs));

        auto res = ComplexTypeMark::mul(context, lhs, rhs, is_lhs_complex, is_rhs_complex);
        if (is_lhs_complex || is_rhs_complex) {
            res = context.mark_node(make_shared<ComplexTypeMark>(res, res.get_element_type()));
        }

        context.mutate_input(0, res);
        return {res};
    } else {
        auto lhs_type = context.get_input_type(0);
        auto rhs_type = context.get_input_type(1);
        // If type is string or None, we shouldn't align
        if (!lhs_type.is<type::Str>() && !rhs_type.is<type::Str>() && !lhs_type.is<type::PyNone>() &&
            !rhs_type.is<type::PyNone>()) {
            align_eltwise_input_types(context,
                                      lhs,
                                      rhs,
                                      is_python_scalar_input(context, 0),
                                      is_python_scalar_input(context, 1));
        }
        auto res = ComplexTypeMark::mul(context, lhs, rhs, is_lhs_complex, is_rhs_complex);
        align_output_types(context, OutputVector{res});
        if (is_lhs_complex || is_rhs_complex) {
            res = context.mark_node(make_shared<ComplexTypeMark>(res, res.get_element_type()));
        }
        return {res};
    }
}

OutputVector translate_mul(const NodeContext& context) {
    return translate_mul_common(context, false);
}

OutputVector translate_mul_(const NodeContext& context) {
    return translate_mul_common(context, true);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
