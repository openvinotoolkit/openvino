// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"

namespace ov::frontend::pytorch::op {

OutputVector translate_index_put(const NodeContext& context) {
    // Pass as PtFrameworkNode to register as `inplace_op`. Conversion to OV operators is done as transformation.
    auto node = std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs());
    return {context.mark_node(node)};
};

}  // namespace ov::frontend::pytorch::op
