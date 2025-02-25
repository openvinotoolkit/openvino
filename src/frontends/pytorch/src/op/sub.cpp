// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
using namespace std;

OutputVector translate_sub_common(const NodeContext& context, bool inplace) {
    num_inputs_check(context, 2, 3, true);
    Output<Node> x = context.get_input(0);
    Output<Node> y = context.get_input(1);

    auto x_complex = ov::as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    auto is_x_complex = (x_complex != nullptr);
    auto y_complex = ov::as_type_ptr<ComplexTypeMark>(y.get_node_shared_ptr());
    auto is_y_complex = (y_complex != nullptr);

    if (x_complex) {
        x = x_complex->input_value(0);
    }
    if (y_complex) {
        y = y_complex->input_value(0);
    }

    if (inplace) {
        if (x.get_element_type().is_dynamic() || x.get_element_type() != y.get_element_type())
            y = context.mark_node(std::make_shared<v1::ConvertLike>(y, x));
    } else {
        align_eltwise_input_types(context,
                                  x,
                                  y,
                                  is_python_scalar_input(context, 0),
                                  is_python_scalar_input(context, 1));
    }

    // default alpha is 1 so no need to multiply if alpha is not provided
    if (!context.input_is_none(2)) {
        auto alpha = context.get_input(2);
        auto alpha_complex = ov::as_type_ptr<ComplexTypeMark>(alpha.get_node_shared_ptr());
        auto is_alpha_complex = (alpha_complex != nullptr);
        alpha = (is_alpha_complex) ? alpha_complex->input_value(0) : alpha;
        auto casted_alpha = context.mark_node(std::make_shared<v1::ConvertLike>(alpha, y));

        y = ComplexTypeMark::mul(context, y, casted_alpha, is_y_complex, is_alpha_complex);
        is_y_complex |= is_alpha_complex;
    }

    auto sub = ComplexTypeMark::sub(context, x, y, is_x_complex, is_y_complex);
    if (is_x_complex || is_y_complex) {
        sub = context.mark_node(make_shared<ComplexTypeMark>(sub, sub.get_element_type()));
    }

    if (inplace)
        context.mutate_input(0, sub);
    return {sub};
};

OutputVector translate_sub(const NodeContext& context) {
    return translate_sub_common(context, false);
};

OutputVector translate_sub_(const NodeContext& context) {
    return translate_sub_common(context, true);
};

OutputVector translate_sub_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    Output<Node> x;
    Output<Node> y;
    std::tie(x, y) = get_inputs_with_promoted_types(context, 0, 1);
    // default alpha is 1 so no need to multiply if alpha is not provided
    if (context.has_attribute("alpha")) {
        auto alpha = context.get_attribute<Output<Node>>("alpha");
        auto casted_alpha = context.mark_node(std::make_shared<v1::ConvertLike>(alpha, y));
        y = context.mark_node(std::make_shared<v1::Multiply>(casted_alpha, y));
    }
    return {context.mark_node(std::make_shared<v1::Subtract>(x, y))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
