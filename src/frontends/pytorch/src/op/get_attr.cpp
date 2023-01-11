// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_get_attr(NodeContext& context) {
    auto res = context.get_decoder()->try_decode_get_attr();
    FRONT_END_OP_CONVERSION_CHECK(res.size() > 0, "GetAttr must have at least one output.");
    return res;
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov