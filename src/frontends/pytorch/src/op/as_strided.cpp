// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/tile.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_as_strided(const NodeContext& context) {
    // "aten::as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)"
    num_inputs_check(context, 3, 4);
    auto input = context.get_input(0);
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto flat_input = context.mark_node(std::make_shared<v1::Reshape>(input, const_neg_1, false));

    std::deque<Output<Node>> sizes;
    std::deque<Output<Node>> strides;
    if (std::dynamic_pointer_cast<v0::Constant>(context.get_input_from_visible_context(1).get_node_shared_ptr())) {
        auto input_vector = context.const_input<std::vector<int64_t>>(1);
        std::for_each(input_vector.rbegin(), input_vector.rend(), [&](int64_t input_val) {
            auto const_input = context.mark_node(v0::Constant::create(element::i32, Shape{}, {input_val}));
            sizes.push_front(const_input);
        });
    } else {
        sizes = get_list_as_outputs(context.get_input(1));
    }
    if (std::dynamic_pointer_cast<v0::Constant>(context.get_input_from_visible_context(2).get_node_shared_ptr())) {
        auto input_vector = context.const_input<std::vector<int64_t>>(2);
        std::for_each(input_vector.rbegin(), input_vector.rend(), [&](int64_t input_val) {
            auto const_input = context.mark_node(v0::Constant::create(element::i32, Shape{}, {input_val}));
            strides.push_front(const_input);
        });
    } else {
        strides = get_list_as_outputs(context.get_input(2));
    }
    auto offset = const_0->output(0);
    if (!context.input_is_none(3)) {
        offset = context.get_input(3);
    }
    FRONT_END_OP_CONVERSION_CHECK(sizes.size() == strides.size(),
                                  "aten::as_strided: Vector for strides and sizes need to have equal length.");
    auto strides_size = strides.size() - 1;
    auto i = 0;
    auto strides_length_const = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {strides.size()}));
    auto ones_strides_len = context.mark_node(std::make_shared<v0::Tile>(const_1, strides_length_const));
    auto indices = const_0;
    std::for_each(strides.rbegin(), strides.rend(), [&](Output<Node>& stride) {
        auto const_num_iter = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {strides_size - i}));
        stride = context.mark_node(std::make_shared<v0::Convert>(stride, element::i32));
        auto size = context.mark_node(std::make_shared<v0::Convert>(sizes.at(strides_size - i), element::i32));
        auto range = context.mark_node(std::make_shared<v4::Range>(const_0, size, const_1, element::i32));
        range = context.mark_node(std::make_shared<v1::Multiply>(range, stride));
        auto iteration_shape = context.mark_node(
            std::make_shared<v3::ScatterUpdate>(ones_strides_len, const_num_iter, const_neg_1, const_0));
        range = context.mark_node(std::make_shared<v1::Reshape>(range, iteration_shape, false));
        indices = context.mark_node(std::make_shared<v1::Add>(indices, range));
        i++;
    });
    indices = context.mark_node(std::make_shared<v1::Add>(indices, offset));
    auto gather = context.mark_node(std::make_shared<v8::Gather>(flat_input, indices, const_0));
    return {gather};
};
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
