// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/hash_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/variable.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_lookup_table_import_op(const NodeContext& node) {
    default_op_checks(node, 3, {"LookupTableImport", "LookupTableImportV2"});
    auto table_handle = node.get_input_by_reference(0);
    auto keys = node.get_input(1);
    auto values = node.get_input(2);
    if (auto ref = as_type_ptr<HashTable>(table_handle.get_node_shared_ptr())) {
        auto new_table = make_shared<HashTable>(*ref, keys, values);

        // since this operation produces new state of the variable
        // it needs to update variables map
        auto variables_state_map = node.get_variable_state_map();
        TENSORFLOW_OP_VALIDATION(node,
                                 variables_state_map,
                                 "[TensorFlow Frontend] internal error: variable state map is nullptr");
        variables_state_map->update_variable_state_map_for_node(node.get_name(), new_table);
    }

    return {};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
