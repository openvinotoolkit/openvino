// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "utils.hpp"

using namespace ov::op;

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
    return {{"add", op::translate_1to1_match_2_inputs<v1::Add>},
            {"broadcast_in_dim", op::translate_broadcast_in_dim},
            {"constant", op::translate_constant},
            {"convert_element_type", op::translate_convert},
            {"conv_general_dilated", op::translate_convolution},
            {"copy", op::skip_node},
            {"device_put", op::skip_node},
            {"div", op::translate_1to1_match_2_inputs<v1::Divide>},
            {"dot_general", op::translate_dot_general},
            {"exp", op::translate_1to1_match_1_input<v0::Exp>},
            {"max", op::translate_1to1_match_2_inputs<v1::Maximum>},
            {"mul", op::translate_1to1_match_2_inputs<v1::Multiply>},
            {"reduce_window_max", op::translate_reduce_window_max},
            {"reduce_window_sum", op::translate_reduce_window_sum},
            {"transpose", op::translate_transpose},
            {"rsqrt", op::translate_rsqrt},
            {"reshape", op::translate_reshape},
            {"slice", op::translate_slice},
            {"sqrt", op::translate_1to1_match_1_input<v0::Sqrt>},
            {"squeeze", op::translate_squeeze},
            {"stop_gradient", op::skip_node},
            {"sub", op::translate_1to1_match_2_inputs<v1::Subtract>},
            {"tanh", op::translate_1to1_match_1_input<v0::Tanh>}};
};

}  // namespace jax
}  // namespace frontend
}  // namespace ov
