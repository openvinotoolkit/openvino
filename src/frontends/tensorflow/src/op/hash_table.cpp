// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow/hash_table.hpp"

#include "common_op_table.hpp"
#include "input_model.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"
#include "variables_index.hpp"

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

    // check maps for HashTable to retrieve keys and values
    // auto model = *(node.get_translate_session())
    auto translate_session = node.get_translate_session();
    TENSORFLOW_OP_VALIDATION(node,
                             translate_session,
                             "[TensorFlow Frontend] internal error: translate session is nullptr.");
    auto model = dynamic_cast<ov::frontend::tensorflow::InputModel*>(translate_session->get_input_model().get());
    TENSORFLOW_OP_VALIDATION(
        node,
        model,
        "[TensorFlow Frontend] internal error: cannot cast a pointer to ov::frontend::tensorflow::InputModel*");
    auto hash_table_keys_map = model->get_hash_table_keys_map();
    auto hash_table_values_map = model->get_hash_table_values_map();

    auto hash_table = make_shared<HashTable>(node_name, key_dtype, value_dtype, node.get_decoder());
    if (hash_table_keys_map.count(node_name) > 0 && hash_table_values_map.count(node_name) > 0) {
        auto keys = hash_table_keys_map.at(node_name)->output(0);
        auto values = hash_table_values_map.at(node_name)->output(0);

        // initialize HashTable since it was found in the map
        auto new_table = make_shared<HashTable>(*hash_table, keys, values);
        return {new_table};
    } else {
        // update variables states of translation session with new unitialized variable
        auto variables_state_map = node.get_variable_state_map();
        TENSORFLOW_OP_VALIDATION(node,
                                 variables_state_map,
                                 "[TensorFlow Frontend] internal error: variable state map is nullptr");
        variables_state_map->update_variable_state_map_for_node(node.get_name(), hash_table);
    }
    return {hash_table};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
