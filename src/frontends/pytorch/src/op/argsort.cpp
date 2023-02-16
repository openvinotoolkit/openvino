// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/topk.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_argsort(NodeContext& context) {
    const auto input_tensor = context.get_input(0);
    const int64_t dim = !context.input_is_none(1) ? context.const_input<int64_t>(1) : -1;
    const bool descending = !context.input_is_none(2) ? context.const_input<bool>(2) : false;
    const auto mode = descending ? TopKMode::MAX : TopKMode::MIN;

    // bool stable{false};
    // if (!context.input_is_none(3)) {
    //     stable = context.const_input<bool>(3);
    // }

    auto shape = context.mark_node(std::make_shared<v0::ShapeOf>(input_tensor));
    ov::Output<ov::Node> output_indices;
    if (dim == -1) {
        ov::Output<ov::Node> shape_copy(shape);
        auto zero_axis = context.mark_node(v0::Constant::create(element::i64, Shape({1}), {0}));
        auto k = context.mark_node(std::make_shared<v1::ReduceProd>(shape_copy, zero_axis, false));
        auto flattened_input_tensor = context.mark_node(std::make_shared<v1::Reshape>(input_tensor, k, false));
        auto flattened_topk =
            context.mark_node(std::make_shared<v3::TopK>(flattened_input_tensor, k, -1, mode, TopKSortType::NONE));
        output_indices = context.mark_node(std::make_shared<v1::Reshape>(flattened_topk->output(1), shape, false));
    } else {
        auto zero_axis = context.mark_node(v0::Constant::create(element::i64, Shape({1}), {0}));
        auto dim_axis = context.mark_node(v0::Constant::create(element::i64, Shape({1}), {dim}));
        auto k = context.mark_node(std::make_shared<v8::Gather>(shape, dim_axis, zero_axis));
        auto topk = context.mark_node(std::make_shared<v3::TopK>(input_tensor, k, dim, mode, TopKSortType::NONE));
        output_indices = topk->output(1);
    }
    auto indices = context.mark_node(std::make_shared<v0::Convert>(output_indices, element::i64));
    return {indices};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov