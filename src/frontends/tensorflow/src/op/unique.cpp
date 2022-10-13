// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/unique.hpp"

#include "op_table.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_unique_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Unique"});
    auto input_values = node.get_input(0);

    // retrieve attribute
    auto output_indices_type = node.get_attribute<ov::element::Type>("out_idx", ov::element::i32);

    auto unique = make_shared<ov::frontend::tensorflow::Unique>(input_values, output_indices_type, node.get_decoder());
    set_node_name(node.get_name(), unique);
    return unique->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
