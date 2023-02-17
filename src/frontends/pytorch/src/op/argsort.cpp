// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/opsets/opset10.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

#define mark(...) context.mark_node(__VA_ARGS__)
#define shared    std::make_shared

using namespace opset10;

OutputVector translate_argsort(NodeContext& context) {
    const auto input_tensor = context.get_input(0);
    const int64_t dim = !context.input_is_none(2) ? context.const_input<int64_t>(2) : -1;
    const bool descending = !context.input_is_none(3) ? context.const_input<bool>(3) : false;
    const bool stable = !context.input_is_none(1) ? context.const_input<bool>(1) : false;

    const auto mode = descending ? ov::op::TopKMode::MAX : ov::op::TopKMode::MIN;
    auto zero_axis = mark(Constant::create(element::i64, Shape({1}), {0}));
    auto dim_axis = mark(Constant::create(element::i64, Shape({1}), {dim}));
    auto shape = mark(shared<ShapeOf>(input_tensor));
    auto elements_node = mark(shared<Gather>(shape, dim_axis, zero_axis));
    auto elements_count = mark(shared<Squeeze>(elements_node));
    auto topk = mark(shared<TopK>(input_tensor, elements_count, dim, mode, ov::op::TopKSortType::NONE));
    auto indices = mark(shared<Convert>(topk->output(1), element::i64));

    return {indices};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
