// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_index(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    if (context.input_is_none(1)) {
        return {x};
    }
    auto indices = context.get_input(1);
    auto index_dtype = context.get_input_type(1);
    if (index_dtype.is<type::List>()) {
        auto list_elems = get_list_as_outputs(indices);
        ov::pass::NodeRegistry rg;
        auto rank = x.get_partial_shape().rank();
        // index transformation supports only tensors with static rank
        PYTORCH_OP_CONVERSION_CHECK(rank.is_static(), "Dynamic rank for aten::index input is not supported.");
        OutputVector ids{list_elems.begin(), list_elems.end()};
        ov::Output<ov::Node> res;
        bool use_input_as_output = true;
        index_tensor_on_list(rg, x, ids, rank.get_length(), res, use_input_as_output);
        context.mark_nodes(rg.get());
        return {res};
    }
    auto index_ov_type = indices.get_element_type();
    if (index_ov_type.is_dynamic()) {
        if (simplified_type_interpret(index_dtype).is<element::Type>()) {
            index_ov_type = index_dtype.as<element::Type>();
        }
    }
    if (index_ov_type == element::boolean || index_ov_type == element::u8) {
        auto nonzero = context.mark_node(std::make_shared<v3::NonZero>(indices, element::i32));
        auto input_order = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, 0}));
        auto masked_id = context.mark_node(std::make_shared<v1::Transpose>(nonzero, input_order));
        auto gather = context.mark_node(std::make_shared<v8::GatherND>(x, masked_id));
        return {gather};
    }
    if (index_ov_type != element::i32) {
        indices = context.mark_node(std::make_shared<ov::op::v0::Convert>(indices, element::i32));
    }
    auto dim = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    return {context.mark_node(std::make_shared<v8::Gather>(x, indices, dim))};
};

OutputVector translate_index_fx(const NodeContext& context) {
    num_inputs_check(context, 2, context.get_input_size());
    auto x = context.get_input(0);
    std::deque<Output<Node>> list_elems;
    for (size_t i = 1; i < context.get_input_size(); i++) {
        Output<Node> index;
        if (!context.input_is_none(i)) {
            index = context.get_input(static_cast<int>(i));
        }
        list_elems.push_back(index);
    }
    ov::pass::NodeRegistry rg;
    auto rank = x.get_partial_shape().rank();
    if (rank.is_dynamic()) {
        rank = context.get_decoder()->get_input_shape(0).rank();
    }
    // index transformation supports only tensors with static rank
    PYTORCH_OP_CONVERSION_CHECK(rank.is_static(), "Dynamic rank for aten::index input is not supported.");

    OutputVector ids{list_elems.begin(), list_elems.end()};
    ov::Output<ov::Node> res;
    bool use_input_as_output = true;
    index_tensor_on_list(rg, x, ids, rank, res, use_input_as_output);
    context.mark_nodes(rg.get());
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
