// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/sigmoid.hpp"
#include "utils.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace jax {
namespace op {

OutputVector translate_logistic(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    auto logistic = std::make_shared<ov::op::v0::Sigmoid>(input);
    return {logistic};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov