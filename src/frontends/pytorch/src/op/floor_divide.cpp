// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_floor_divide(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    align_eltwise_input_types(context, x, y, true);
    auto div = context.mark_node(std::make_shared<v1::Divide>(x, y, true));
    auto floor = context.mark_node(std::make_shared<v0::Floor>(div));
    auto out_type = context.get_output_type(0);
    if (out_type.is<element::Type>()) {
        floor = context.mark_node(std::make_shared<v0::Convert>(floor, out_type.as<element::Type>()));
    }
    return {floor};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov