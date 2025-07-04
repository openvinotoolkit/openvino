// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_size(const NodeContext& context) {
    num_inputs_check(context, 1, 2, true);
    auto data = context.get_input(0);
    Output<Node> shape;

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(data.get_node_shared_ptr());
    if (complex_type_mark) {
        data = complex_type_mark->get_data();
        shape = context.mark_node(std::make_shared<v3::ShapeOf>(data, element::i64));

        auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
        auto stop = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
        auto step = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
        shape = context.mark_node(std::make_shared<v8::Slice>(shape, zero, stop, step, zero));
    } else {
        shape = context.mark_node(std::make_shared<v3::ShapeOf>(data, element::i64));
    }

    if (context.input_is_none(1)) {
        return {shape};
    } else {
        auto axis_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
        return {context.mark_node(std::make_shared<v8::Gather>(shape, context.get_input(1), axis_0))};
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
