// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/hash_table.hpp"

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_hash_table_op(const ov::frontend::tensorflow::NodeContext& node) {
    default_op_checks(node, 0, {"MutableHashTable", "MutableHashTableV2", "HashTable", "HashTableV2"});

    auto hash_table = make_shared<HashTable>(node.get_decoder());
    set_node_name(node.get_name(), hash_table);
    return {hash_table};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
