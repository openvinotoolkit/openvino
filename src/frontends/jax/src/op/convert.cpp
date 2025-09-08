// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert.hpp"

#include <cstdint>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

OutputVector translate_convert(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    Output<Node> src = context.get_input(0);
    auto dtype = convert_dtype(context.const_named_param<int64_t>("new_dtype"));
    Output<Node> res = std::make_shared<v0::Convert>(src, dtype);
    return {res};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov
