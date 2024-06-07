// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

#define OP_CONVERTER(op) OutputVector op(const NodeContext &node)

OP_CONVERTER(translate_constant);

} // namespace op

// Supported ops for Jaxpr
const std::map<std::string, CreatorFunction> get_supported_ops_jaxpr() {
  return {{"add", op::translate_1to1_match_2_inputs<opset10::Add>},
          {"constant", op::translate_constant}};
};

} // namespace jax
} // namespace frontend
} // namespace ov
