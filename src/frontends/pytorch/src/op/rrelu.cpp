// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/prelu.hpp"
#include "utils.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/add.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_rrelu(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);

    Output<Node> negative_slope = ov::op::v0::Constant::create(x.get_element_type(), Shape{1}, {11.0 / 48.0});
    if (!context.input_is_none(1) && !context.input_is_none(2)) {
        Output<Node> lower = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(context.get_input(1), x));
        Output<Node> upper = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(context.get_input(2), x));
        negative_slope = context.mark_node(std::make_shared<ov::op::v1::Divide>(
            std::make_shared<ov::op::v1::Add>(lower, upper),
            ov::op::v0::Constant::create(x.get_element_type(), Shape{1}, {2.0})
        ));
    }
    return {context.mark_node(std::make_shared<v0::PRelu>(x, negative_slope))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
