// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hash_table.hpp"

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "utils.hpp"

using namespace ov;
using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_hash_table_op(const ov::frontend::tensorflow::NodeContext& node) {
    default_op_checks(node, 0, {"MutableHashTable", "MutableHashTableV2", "HashTable", "HashTableV2"});
    auto node_name = node.get_name();
    auto key_dtype = node.get_attribute<element::Type>("key_dtype");
    auto value_dtype = node.get_attribute<element::Type>("value_dtype");

    auto hash_table = make_shared<HashTable>(node_name, key_dtype, value_dtype, node.get_decoder());
    return {hash_table};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
