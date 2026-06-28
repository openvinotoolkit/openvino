// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_delete(const NodeContext& context) {
    // aten::Delete.t(t[](a!) self, int idx) -> ()
    // Removes element `idx` from a list represented as a tensor stacked along axis 0
    // (the representation aten::_set_item operates on). Result = gather of every
    // index except `idx`, propagated back to the list via mutate_input. A simple
    // no-op would be wrong: subsequent len()/indexed reads must see the element gone.
    num_inputs_check(context, 2, 2);
    auto self = context.get_input(0);
    auto idx = context.get_input(1);

    auto const_0 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));
    auto scalar_shape = context.mark_node(v0::Constant::create(element::i64, Shape{0}, std::vector<int64_t>{}));

    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(self, element::i64));
    auto len = context.mark_node(std::make_shared<v8::Gather>(shape, const_0, const_0));  // scalar dim-0 length

    idx = context.mark_node(std::make_shared<v0::Convert>(idx, element::i64));
    idx = context.mark_node(std::make_shared<v1::Reshape>(idx, scalar_shape, false));     // force scalar
    idx = context.mark_node(std::make_shared<v1::FloorMod>(idx, len));                     // python negative-index

    auto idx_plus_1 = context.mark_node(std::make_shared<v1::Add>(idx, const_1));
    auto first = context.mark_node(std::make_shared<v4::Range>(const_0, idx, const_1, element::i64));      // [0, idx)
    auto second = context.mark_node(std::make_shared<v4::Range>(idx_plus_1, len, const_1, element::i64));  // [idx+1, len)
    auto kept = context.mark_node(std::make_shared<v0::Concat>(OutputVector{first, second}, 0));

    auto result = context.mark_node(std::make_shared<v8::Gather>(self, kept, const_0));
    context.mutate_input(0, result);
    return {};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
