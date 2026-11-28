// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <limits>
#include <functional>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/frontend/sequence_erase.hpp"
#include "openvino/frontend/sequence_insert.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

OutputVector translate_reverse(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    Output<Node> result;
    if (const auto sequence = ov::as_type_ptr<SequenceMark>(input.get_node_shared_ptr())) {
        std::function<bool(const Output<Node>&)> is_static_sequence = [&](const Output<Node>& value) {
            const auto node = value.get_node_shared_ptr();
            if (const auto mark = ov::as_type_ptr<SequenceMark>(node)) {
                for (const auto& item : mark->input_values()) {
                    if ((ov::is_type<SequenceInsert>(item.get_node()) || ov::is_type<SequenceErase>(item.get_node())) &&
                        !is_static_sequence(item)) {
                        return false;
                    }
                }
                return true;
            }
            if (const auto insert = ov::as_type_ptr<SequenceInsert>(node)) {
                return is_static_sequence(insert->get_input_sequence());
            }
            if (const auto erase = ov::as_type_ptr<SequenceErase>(node)) {
                return is_static_sequence(erase->get_input_sequence());
            }
            return false;
        };
        PYTORCH_OP_CONVERSION_CHECK(is_static_sequence(input), "Cannot reverse a dynamically sized tensor list.");
        auto elements = sequence->get_sequence();
        for (const auto& element : elements) {
            PYTORCH_OP_CONVERSION_CHECK(!ov::is_type<SequenceInsert>(element.get_node()) &&
                                            !ov::is_type<SequenceErase>(element.get_node()),
                                        "Cannot reverse a list with dynamic insertions or removals.");
        }
        std::reverse(elements.begin(), elements.end());
        result = context.mark_node(make_list_construct(elements));
    } else {
        const auto input_type = context.get_input_type(0);
        PYTORCH_OP_CONVERSION_CHECK(!input_type.is<type::List>() ||
                                        !input_type.as<type::List>().element_type.is<type::Tensor>(),
                                    "Cannot reverse a dynamically sized tensor list.");
        PYTORCH_OP_CONVERSION_CHECK(!ov::is_type<ov::op::util::FrameworkNode>(input.get_node()),
                                    "Cannot reverse an unresolved list.");
        auto start = context.mark_node(ov::op::v0::Constant::create(element::i64, Shape{1}, {-1}));
        auto stop = context.mark_node(ov::op::v0::Constant::create(
            element::i64, Shape{1}, {std::numeric_limits<int64_t>::min()}));
        auto step = context.mark_node(ov::op::v0::Constant::create(element::i64, Shape{1}, {-1}));
        result = context.mark_node(std::make_shared<ov::op::v8::Slice>(input, start, stop, step));
    }
    context.mutate_input(0, result);
    return {result};
}

}  // namespace ov::frontend::pytorch::op
