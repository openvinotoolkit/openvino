// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include "openvino/opsets/opset14.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

#define OP_CONVERTER(op) OutputVector op(const NodeContext& node)

OP_CONVERTER(translate_constant);
OP_CONVERTER(translate_convert);
OP_CONVERTER(translate_convolution);
OP_CONVERTER(translate_reduce_window_max);

}  // namespace op

// Supported ops for Jaxpr
const std::map<std::string, CreatorFunction> get_supported_ops_jaxpr() {
    return {{"add", op::translate_1to1_match_2_inputs<opset14::Add>},
            {"sub", op::translate_1to1_match_2_inputs<opset14::Subtract>},
            {"mul", op::translate_1to1_match_2_inputs<opset14::Multiply>},
            {"div", op::translate_1to1_match_2_inputs<opset14::Divide>},
            {"constant", op::translate_constant},
            {"convert_element_type", op::translate_convert},
            {"conv_general_dilated", op::translate_convolution},
            {"reduce_window_max", op::translate_reduce_window_max}};
};

}  // namespace jax
}  // namespace frontend
}  // namespace ov
