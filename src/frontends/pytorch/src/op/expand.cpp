// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector base_expand(const NodeContext& context, const Output<Node>& x, const Output<Node>& sizes) {
    auto shape = context.mark_node(std::make_shared<v0::Abs>(sizes));
    return {context.mark_node(std::make_shared<v3::Broadcast>(x, shape, BroadcastType::BIDIRECTIONAL))};
};
}  // namespace

OutputVector translate_expand(const NodeContext& context) {
    // aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto sizes = context.get_input(1);
    // TODO: figure out what implicit means
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(2) || context.const_input<bool>(2) == false,
                                "Unexpected value of implicit for expand operation");
    return base_expand(context, x, sizes);
};

OutputVector translate_expand_as(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(y, element::i32));
    return {context.mark_node(std::make_shared<v3::Broadcast>(x, shape, BroadcastType::BIDIRECTIONAL))};
};

OutputVector translate_expand_fx(const NodeContext& context) {
    auto num_inputs = context.get_input_size();
    num_inputs_check(context, 2, num_inputs);
    auto x = context.get_input(0);
    std::vector<int32_t> shape_vec;
    if (context.get_input_type(1).is<type::List>()) {
        std::deque<Output<Node>> list_elems;
        for (size_t i = 1; i < num_inputs; i++) {
            if (context.get_input_type(i).as<type::List>().element_type.is<type::PyScalar>()) {
                auto const_val = context.const_input<int32_t>(i);
                std::vector<int32_t> dim_vec;
                dim_vec.push_back(const_val);
                auto dim_const = ov::op::v0::Constant::create(element::i32, Shape{1}, dim_vec);
                list_elems.push_back(dim_const);
            } else {
                auto converted_dim = context.mark_node(
                    std::make_shared<ov::op::v0::Convert>(context.get_input(static_cast<int>(i)), element::i32));
                if (converted_dim->get_output_partial_shape(0).rank() == 0) {
                    auto dims_1d_shape = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{1}, {-1}));
                    auto reshape_dim =
                        context.mark_node(std::make_shared<ov::op::v1::Reshape>(converted_dim, dims_1d_shape, false));
                    list_elems.push_back(reshape_dim);
                } else {
                    list_elems.push_back(converted_dim);
                }
            }
        }
        auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector(list_elems.begin(), list_elems.end()), 0);
        return base_expand(context, x, concat);
    } else {
        auto x = context.get_input(0);
        auto sizes = context.get_input(1);
        // TODO: figure out what implicit means
        PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(2) || context.const_input<bool>(2) == false,
                                    "Unexpected value of implicit for expand operation");
        return base_expand(context, x, sizes);
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
