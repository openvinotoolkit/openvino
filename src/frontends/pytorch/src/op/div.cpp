// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "transformations/rt_info/nonconvertible_divide.hpp"
#include "utils.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

namespace {
OutputVector translate_div_common(const NodeContext& context,
                                  const Output<Node>& lhs,
                                  const Output<Node>& rhs,
                                  const std::string& rounding_mode,
                                  bool inplace) {
    auto x = lhs;
    auto y = rhs;
    if (rounding_mode.empty()) {
        // if no rounding mode and both inputs are ints cast BOTH to fp32
        const auto x_dtype = x.get_element_type();
        const auto y_dtype = y.get_element_type();
        if (x_dtype.is_static() && x_dtype.is_integral() && y_dtype.is_static() && y_dtype.is_integral()) {
            x = context.mark_node(std::make_shared<v0::Convert>(x, element::f32));
            y = context.mark_node(std::make_shared<v0::Convert>(y, element::f32));
        }
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
    auto res = context.mark_node(std::make_shared<v1::Divide>(x, y, true));
    // TODO: ticket 103296; Temporarily disable ConvertDivide transformation
    disable_divide_conversion(res);
    if (rounding_mode == "floor") {
        res = context.mark_node(std::make_shared<v0::Floor>(res));
    } else if (rounding_mode == "trunc") {
        const auto convert = context.mark_node(std::make_shared<v0::Convert>(res, element::i32));
        res = context.mark_node(std::make_shared<v1::ConvertLike>(convert, x));
    }
    if (inplace)
        context.mutate_input(0, res);
    return {res};
};
}  // namespace

OutputVector translate_div(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    std::string rounding_mode = "";
    if (!context.input_is_none(2)) {
        rounding_mode = context.const_input<std::string>(2);
    }
    return translate_div_common(context, x, y, rounding_mode, false);
};

OutputVector translate_div_(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    std::string rounding_mode = "";
    if (!context.input_is_none(2)) {
        rounding_mode = context.const_input<std::string>(2);
    }
    return translate_div_common(context, x, y, rounding_mode, true);
};

OutputVector translate_div_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    std::string rounding_mode = "";
    if (context.has_attribute("rounding_mode")) {
        rounding_mode = context.get_attribute<std::string>("rounding_mode");
    }
    return translate_div_common(context, x, y, rounding_mode, false);
};

OutputVector translate_div_fx_(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    std::string rounding_mode = "";
    if (context.has_attribute("rounding_mode")) {
        rounding_mode = context.get_attribute<std::string>("rounding_mode");
    }
    return translate_div_common(context, x, y, rounding_mode, true);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
