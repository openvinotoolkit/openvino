// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_lookup_table_insert_op(const ov::frontend::tensorflow::NodeContext& node) {
    // auto-pruning of unsupported sub-graphs that contain
    // operations working with dictionaries
    default_op_checks(node, 3, {"LookupTableInsert", "LookupTableInsertV2"});
    return {};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
