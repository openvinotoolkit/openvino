// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
OutputVector translate_max_unpool2d(const NodeContext& context) {
    num_inputs_check(context, 5, 6);
    auto input = context.get_input(0);
    auto indices = context.get_input(1);
    bool has_output_size = context.get_input_size() == 6;

    auto expand_2d_param = [](const std::vector<int64_t>& param) {
        if (param.size() == 1) return std::vector<int64_t>{param[0], param[0]};
        return param;
    };

    auto get_const_vector = [&](const Output<Node>& input) -> std::vector<int64_t> {
        auto const_node = ov::as_type_ptr<ov::op::v0::Constant>(input.get_node_shared_ptr());
        if (!const_node) {
            FRONT_END_THROW("Expected constant input for kernel/stride/padding.");
        }
        return const_node->cast_vector<int64_t>();
    };

    auto make_const = [&](int64_t val) { return v0::Constant::create(element::i64, {}, {val}); };

    auto kernel = expand_2d_param(get_const_vector(context.get_input(2)));
    auto stride = expand_2d_param(get_const_vector(context.get_input(3)));
    auto padding = expand_2d_param(get_const_vector(context.get_input(4)));

    Output<Node> output_shape;
    Output<Node> chw, hw, w;

    if (has_output_size) {
        output_shape = context.get_input(5);
    } else {
        auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i64));
        auto batch_size = context.mark_node(std::make_shared<v1::Gather>(input_shape, make_const(0), make_const(0)));
        auto channels = context.mark_node(std::make_shared<v1::Gather>(input_shape, make_const(1), make_const(0)));
        auto height = context.mark_node(std::make_shared<v1::Gather>(input_shape, make_const(2), make_const(0)));
        auto width = context.mark_node(std::make_shared<v1::Gather>(input_shape, make_const(3), make_const(0)));

        auto calculate_newdim = [&](Output<Node> dim, int64_t k, int64_t s, int64_t p) {
            auto sub = context.mark_node(std::make_shared<v1::Subtract>(dim, make_const(1)));
            auto mul = context.mark_node(std::make_shared<v1::Multiply>(sub, make_const(s)));
            return context.mark_node(std::make_shared<v1::Add>(mul, make_const(k - 2 * p)));
        };

        auto new_height = calculate_newdim(height, kernel[0], stride[0], padding[0]);
        auto new_width = calculate_newdim(width, kernel[1], stride[1], padding[1]);
        w = new_width;
        hw = context.mark_node(std::make_shared<v1::Multiply>(new_height, new_width));
        chw = context.mark_node(std::make_shared<v1::Multiply>(channels, hw));

        output_shape = context.mark_node(std::make_shared<v0::Concat>(
            NodeVector{batch_size, channels, new_height, new_width}, 0));
    }

    auto flat_input = context.mark_node(std::make_shared<v1::Reshape>(input, make_const(-1), false));
    auto flat_indices = context.mark_node(std::make_shared<v1::Reshape>(indices, make_const(-1), false));

    auto new_batch = context.mark_node(std::make_shared<v1::Divide>(flat_indices, chw));
    auto rem_1 = context.mark_node(std::make_shared<v1::Mod>(flat_indices, chw));
    auto new_channel = context.mark_node(std::make_shared<v1::Divide>(rem_1, hw));
    auto rem_2 = context.mark_node(std::make_shared<v1::Mod>(rem_1, hw));
    auto new_height = context.mark_node(std::make_shared<v1::Divide>(rem_2, w));
    auto new_width = context.mark_node(std::make_shared<v1::Mod>(rem_2, w));

    auto unsqueeze = [&](Output<Node> n) {
        return context.mark_node(std::make_shared<v0::Unsqueeze>(n, make_const(1)));
    };

    auto new_indices = context.mark_node(std::make_shared<v0::Concat>(
        NodeVector{
            unsqueeze(new_batch),
            unsqueeze(new_channel),
            unsqueeze(new_height),
            unsqueeze(new_width)},
        1));

    auto zero_tensor = context.mark_node(std::make_shared<v3::Broadcast>(make_const(0), output_shape));
    auto unpooled = context.mark_node(std::make_shared<v3::ScatterNDUpdate>(
        zero_tensor, new_indices, flat_input));

    return {unpooled};
}



}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov