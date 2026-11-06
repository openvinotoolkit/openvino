// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_pop(const NodeContext& context) {
    // aten::pop.t(t[](a!) self, int idx=-1) -> t
    num_inputs_check(context, 1, 2);
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(1) || !context.get_input_type(1).is<type::Str>(),
                                "String key in aten::pop means dict input, this is not supported.");
    int64_t idx = -1;
    if (!context.input_is_none(1)) {
        idx = context.const_input<int64_t>(1);
    }
    auto input = context.get_input(0);

    if (!ov::as_type_ptr<ov::op::util::FrameworkNode>(input.get_node_shared_ptr())) {
        // The list is represented as a 1D tensor, for example a list of ints produced from a shape.
        auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
        auto one = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
        auto axis = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
        auto idx_const = context.mark_node(v0::Constant::create(element::i64, Shape{}, {idx}));
        auto popped = context.mark_node(std::make_shared<v8::Gather>(input, idx_const, zero));

        auto start = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
        auto head_end = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {idx}));
        auto head = context.mark_node(std::make_shared<v8::Slice>(input, start, head_end, one, axis));
        // For idx == -1 the tail is empty, so it starts past the end of the list.
        const int64_t tail_start_value = idx == -1 ? std::numeric_limits<int64_t>::max() : idx + 1;
        auto tail_start = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {tail_start_value}));
        auto tail_end =
            context.mark_node(v0::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()}));
        auto tail = context.mark_node(std::make_shared<v8::Slice>(input, tail_start, tail_end, one, axis));
        auto rest = context.mark_node(std::make_shared<v0::Concat>(OutputVector{head, tail}, 0));
        // aten::pop removes the element from the input list in-place.
        context.mutate_input(0, rest);
        return {popped};
    }

    auto list_elems = get_list_as_outputs(input);
    const auto list_size = static_cast<int64_t>(list_elems.size());
    if (idx < 0) {
        idx += list_size;
    }
    PYTORCH_OP_CONVERSION_CHECK(idx >= 0 && idx < list_size,
                                "Index: ",
                                idx,
                                " is out of bounds of input list of len: ",
                                list_size);
    const auto popped = list_elems.at(idx);
    list_elems.erase(list_elems.begin() + idx);
    auto result = context.mark_node(std::make_shared<SequenceMark>(OutputVector(list_elems.begin(), list_elems.end())));
    // aten::pop removes the element from the input list in-place.
    context.mutate_input(0, result);
    return {popped};
}

}  // namespace ov::frontend::pytorch::op
