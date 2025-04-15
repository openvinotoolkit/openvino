// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_index_put(const NodeContext& context) {
    // Pass as PtFrameworkNode to register as `inplace_op`. Conversion to OV operators is done as transformation.
    auto node = std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs());
    return {context.mark_node(node)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
