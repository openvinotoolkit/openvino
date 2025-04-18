// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_poisson(const NodeContext& context) {
    
    num_inputs_check(context, 1, 2);
    auto rates = context.get_input(0);
    
    if (!context.input_is_none(1)) {
        PYTORCH_OP_CONVERSION_CHECK(false, "aten::poisson conversion with generator is not supported");
    }

    auto fw_node = std::make_shared<PtFrameworkNode>(context.get_decoder(), OutputVector{rates}, 1);
    fw_node->set_output_type(0, rates.get_element_type(), rates.get_partial_shape());
    auto res = context.mark_node(fw_node);
    return {res};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov 