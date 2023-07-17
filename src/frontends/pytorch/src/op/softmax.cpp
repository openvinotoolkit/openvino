// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softmax.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
OutputVector translate_softmax(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto axis = context.const_input<int64_t>(1);
    if (!context.input_is_none(2)) {
        x = apply_dtype(context, 2, x);
    }
    return {context.mark_node(std::make_shared<v8::Softmax>(x, axis))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov