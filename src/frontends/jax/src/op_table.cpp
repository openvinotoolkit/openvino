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

OP_CONVERTER(translate_broadcast_in_dim);
OP_CONVERTER(translate_constant);
OP_CONVERTER(translate_convert);
OP_CONVERTER(translate_convolution);
OP_CONVERTER(translate_copy);
OP_CONVERTER(translate_dot_general);
OP_CONVERTER(translate_reduce_window_max);
OP_CONVERTER(translate_reduce_window_sum);
OP_CONVERTER(translate_reshape);
OP_CONVERTER(translate_rsqrt);
OP_CONVERTER(translate_slice);
OP_CONVERTER(translate_squeeze);
OP_CONVERTER(translate_transpose);

}  // namespace op

// Supported ops for Jaxpr
const std::map<std::string, CreatorFunction> get_supported_ops_jaxpr() {
    return {{"add", op::translate_1to1_match_2_inputs<opset14::Add>},
            {"sub", op::translate_1to1_match_2_inputs<opset14::Subtract>},
            {"mul", op::translate_1to1_match_2_inputs<opset14::Multiply>},
            {"div", op::translate_1to1_match_2_inputs<opset14::Divide>},
            {"broadcast_in_dim", op::translate_broadcast_in_dim},
            {"constant", op::translate_constant},
            {"convert_element_type", op::translate_convert},
            {"conv_general_dilated", op::translate_convolution},
            {"copy", op::skip_node},
            {"dot_general", op::translate_dot_general},
            {"max", op::translate_1to1_match_2_inputs<opset14::Maximum>},
            {"reduce_window_max", op::translate_reduce_window_max},
            {"reduce_window_sum", op::translate_reduce_window_sum},
            {"transpose", op::translate_transpose},
            {"rsqrt", op::translate_rsqrt},
            {"reshape", op::translate_reshape},
            {"slice", op::translate_slice},
            {"squeeze", op::translate_squeeze}};
};

}  // namespace jax
}  // namespace frontend
}  // namespace ov
