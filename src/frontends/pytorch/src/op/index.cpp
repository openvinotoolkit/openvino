// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather_nd.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_index_fx(const NodeContext& context) {
    num_inputs_check(context, 2, context.get_input_size());
    auto x = context.get_input(0);
    std::deque<Output<Node>> list_elems;
    for (size_t i = 1; i < context.get_input_size(); i++) {
        auto index = context.get_input(static_cast<int>(i));
        if (index.get_element_type() == element::i64) {
            auto converted = context.mark_node(std::make_shared<ov::op::v0::Convert>(index, element::i32));
            list_elems.push_back(converted);
        } else {
            list_elems.push_back(index);
        }
    }
    auto concat =
        context.mark_node(std::make_shared<ov::op::v0::Concat>(OutputVector(list_elems.begin(), list_elems.end()), 0));
    auto gather = std::make_shared<ov::op::v8::GatherND>(x, concat);
    return {gather};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
