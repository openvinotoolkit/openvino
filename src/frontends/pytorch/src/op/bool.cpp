// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_bool(NodeContext& context) {
    return {context.mark_node(std::make_shared<ov::op::v0::Convert>(context.get_input(0), element::boolean))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
