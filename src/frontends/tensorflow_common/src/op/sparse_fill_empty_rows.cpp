// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_fill_empty_rows.hpp"

#include "common_op_table.hpp"
#include "openvino/op/convert.hpp"
#include "tf_framework_node.hpp"
#include "utils.hpp"

namespace ov::frontend::tensorflow::op {
NamedOutputVector translate_sparse_fill_empty_rows_op(const NodeContext& node) {
    default_op_checks(node, 4, {"SparseFillEmptyRows"});
    auto indices = node.get_input(0);
    auto values = node.get_input(1);
    auto dense_shape = node.get_input(2);
    auto default_value = node.get_input(3);
    TENSORFLOW_OP_VALIDATION(node,
                             values.get_element_type() != element::string,
                             "TensorFlow Frontend does not support string inputs for SparseFillEmptyRows.");

    auto sparse_fill_empty_rows =
        std::make_shared<ov::op::v16::SparseFillEmptyRows>(values, dense_shape, indices, default_value);

    set_node_name(node.get_name(), sparse_fill_empty_rows);
    auto outputs = sparse_fill_empty_rows->outputs();

    return {{"output_indices", outputs[0]}, {"output_values", outputs[1]}, {"empty_row_indicator", outputs[2]}};
}
}  // namespace ov::frontend::tensorflow::op
