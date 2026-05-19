// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/col2im.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_col2im(const NodeContext& context) {
    num_inputs_check(context, 3, 7);
    auto x = context.get_input(0);
    auto output_size = get_input_concat_if_list(context, 1);
    auto kernel_size = get_input_concat_if_list(context, 2);
    auto dilations = context.const_input<Strides>(3);
    auto padding = context.const_input<Shape>(4);
    auto strides = context.const_input<Strides>(5);

    auto col2im = context.mark_node(
        std::make_shared<v15::Col2Im>(x, output_size, kernel_size, strides, dilations, padding, padding));
    return {col2im};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov