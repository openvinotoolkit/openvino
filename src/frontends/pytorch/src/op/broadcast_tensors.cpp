// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_broadcast_tensors(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto list_tensors = context.get_input(0).get_node_shared_ptr();
    if (cast_fw_node(list_tensors, "prim::ListConstruct")) {
        auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
        auto final_shape_t = context.mark_node(std::make_shared<v1::ConvertLike>(zero, list_tensors->get_input_source_output(0)));
        for (size_t i; i < list_tensors->get_input_size(); i++){
            final_shape_t =
                context.mark_node(std::make_shared<v1::Add>(final_shape_t, list_tensors->get_input_source_output(i)));
        }
        auto final_shape = context.mark_node(std::make_shared<v3::ShapeOf>(final_shape_t, element::i32));
        final_shape = context.mark_node(std::make_shared<v0::Unsqueeze>(final_shape, zero));
        OutputVector output_tensors;
        for (size_t i; i < list_tensors->get_input_size(); i++){
            auto broadcasted_tensor = context.mark_node(
                std::make_shared<v3::Broadcast>(list_tensors->get_input_source_output(i), final_shape));
            output_tensors.push_back(broadcasted_tensor);
        }

        auto out_concat = context.mark_node(std::make_shared<v0::Concat>(output_tensors, 0));
    }
    return {list_tensors};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
