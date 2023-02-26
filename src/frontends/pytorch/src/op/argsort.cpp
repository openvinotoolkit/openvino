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
    bool stable;
    int64_t dim;
    bool descending;

    if (context.get_input_size() == 4) {
        stable = context.const_input<bool>(1);
        FRONT_END_OP_CONVERSION_CHECK(stable == false, "Stable sorting in aten::argsort is not yet supported.");
        dim = context.const_input<int64_t>(2);
        descending = context.const_input<bool>(3);
    } else {
        dim = context.const_input<int64_t>(1);
        descending = context.const_input<bool>(2);
    }
    auto mode = descending ? ov::op::TopKMode::MAX : ov::op::TopKMode::MIN;

    auto zero_axis = mark(Constant::create(element::i64, Shape({1}), {0}));
    auto dim_axis = mark(Constant::create(element::i64, Shape({1}), {dim}));
    auto shape = mark(shared<ShapeOf>(input_tensor));
    auto k_values_node = mark(shared<Gather>(shape, dim_axis, zero_axis));
    auto k_values = mark(shared<Squeeze>(k_values_node));
    auto topk = mark(shared<TopK>(input_tensor, k_values, dim, mode, ov::op::TopKSortType::NONE));
    auto indices = mark(shared<Convert>(topk->output(1), element::i64));

    return {indices};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
