// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_randperm(const NodeContext& context) {
    auto num_inputs = context.get_input_size();
    auto n_node = context.get_input(0);
    int dtype_value = 4;
    if (num_inputs == 1) {
    } else if (num_inputs == 2) {
        if (!context.input_is_none(1)) {
            dtype_value = context.const_input<int>(1);
            PYTORCH_OP_CONVERSION_CHECK(dtype_value == 4,
                                        "Only dtype value 4 (int64) is supported for aten::randperm, got: ",
                                        dtype_value);
        }
    } else if (num_inputs == 5) {
        if (!context.input_is_none(1)) {
            dtype_value = context.const_input<int>(1);
            PYTORCH_OP_CONVERSION_CHECK(dtype_value == 4,
                                        "Only dtype value 4 (int64) is supported for aten::randperm, got: ",
                                        dtype_value);
        }
    } else {
        PYTORCH_OP_CONVERSION_CHECK(false, "Unexpected number of inputs for aten::randperm: ", num_inputs);
    }
    auto axis_zero = v0::Constant::create(element::i64, Shape{1}, {0});
    auto shape = context.mark_node(std::make_shared<v0::Unsqueeze>(n_node, axis_zero));
    auto min_val = v0::Constant::create(element::f32, Shape{}, {0.0f});
    auto max_val = v0::Constant::create(element::f32, Shape{}, {1.0f});
    auto random_tensor = context.mark_node(std::make_shared<v8::RandomUniform>(shape, min_val, max_val, element::f32));
    const int64_t axis = 0;
    auto topk = context.mark_node(std::make_shared<v11::TopK>(random_tensor,
                                                              n_node,
                                                              axis,
                                                              ov::op::TopKMode::MIN,
                                                              ov::op::TopKSortType::SORT_VALUES,
                                                              element::i64,
                                                              false));
    return {topk->output(1)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
