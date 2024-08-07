// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/col2im.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_col2im(const NodeContext& context) {
    num_inputs_check(context, 3, 7);
    auto x = context.get_input(0);
    auto kernel_size = context.get_input(2);
    auto dilations = context.const_input<Strides>(3);
    auto padding = context.const_input<Shape>(4);
    auto strides = context.const_input<Strides>(5);

    Output<Node> output_size;
    auto shape_type = context.get_input_type(1);
    if (shape_type.is<type::List>()) {
        const auto list_elems = get_list_as_outputs(context.get_input(1));
        if (list_elems.size() == 1) {
            output_size = list_elems[0];
        } else {
            OutputVector to_concat;
            auto zero = v0::Constant::create(element::i32, Shape{}, {0});
            for (auto elem : list_elems) {
                to_concat.push_back(context.mark_node(std::make_shared<v0::Unsqueeze>(elem, zero)));
            }
            output_size = context.mark_node(std::make_shared<v0::Concat>(to_concat, 0));
        }
    } else {
        output_size = context.get_input(1);
    }

    auto col2im = context.mark_node(
        std::make_shared<v15::Col2Im>(x, output_size, kernel_size, strides, dilations, padding, padding));
    return {col2im};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov