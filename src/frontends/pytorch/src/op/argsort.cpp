// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"


namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_argsort(NodeContext& context) {
    auto const input_tensor = context.get_input(0);

    int64_t dim{-1};
    if(!context.input_is_none(1)) {
        dim = context.const_input<int>(1);
    }
    
    bool descending{false};
    if(!context.input_is_none(2)) {
        descending = context.const_input<bool>(2);
    }
    auto mode = descending ? TopKMode::MAX : TopKMode::MIN;

    bool stable{false};
    if(!context.input_is_none(3)) {
        stable = context.const_input<bool>(3);
    }

    auto shape = context.mark_node(std::make_shared<v0::ShapeOf>(input_tensor));

    ov::Output<ov::Node> output_indices;
    if(dim == -1) {
        auto k = numel(context, shape);
        auto flattened_input_tensor = context.mark_node(std::make_shared<v1::Reshape>(input_tensor, k, false));
        auto flattened_topk = context.mark_node(std::make_shared<v3::TopK>(flattened_input_tensor, k, -1, mode, TopKSortType::NONE));
        output_indices = context.mark_node(std::make_shared<v1::Reshape>(flattened_topk->output(1), shape));
    } else {
        auto k = context.mark_node(std::make_shared<v8::Gather>(shape, Shape{(uint64_t)dim}, Shape{0}));
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