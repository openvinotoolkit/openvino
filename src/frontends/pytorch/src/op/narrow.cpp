// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_narrow(const NodeContext& context) {
    num_inputs_check(context, 4, 4, true);  // allow_complex = true

    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto [data, complex] = unwrap_complex(context.get_input(0));
    auto dim = context.get_input(1);
    auto start_input = context.get_input(2);
    auto length = context.get_input(3);

    if (complex) {
        if (dim.get_element_type() != element::i32) {
            dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
        }
        auto rank = std::get<1>(get_shape_rank(context, context.get_input(0), true));
        dim = normalize_axis(context, dim, rank);
    }

    auto start = context.mark_node(std::make_shared<v0::Unsqueeze>(start_input, const_0));
    auto stop = context.mark_node(std::make_shared<v1::Add>(start, length));
    auto axis = context.mark_node(std::make_shared<v0::Unsqueeze>(dim, const_0));

    auto narrow = context.mark_node(std::make_shared<v8::Slice>(data, start, stop, const_1, axis));
    return {wrap_complex(context, narrow, complex)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov