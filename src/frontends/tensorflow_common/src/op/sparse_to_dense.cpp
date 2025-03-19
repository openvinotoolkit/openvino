// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_sparse_to_dense_op(const NodeContext& node) {
    default_op_checks(node, 3, {"SparseToDense"});
    // This replacer substitutes TensorFlow SparseToDense operation with Broadcast -> ScatterND chain.
    // The Broadcast operation creates a tensor filled with default value and of required shape.
    // The ScatterND operation updates the created tensor with required values at required locations.
    auto input_size = node.get_input_size();
    TENSORFLOW_OP_VALIDATION(node,
                             input_size == 3 || input_size == 4,
                             "SparseToDense must have either three or four inputs.");

    auto indices = node.get_input(0);
    auto dense_shape = node.get_input(1);
    auto values = node.get_input(2);
    shared_ptr<v3::Broadcast> broadcast = nullptr;
    if (input_size == 3) {
        auto const_zero = create_same_type_const_scalar<int32_t>(values, 0);
        broadcast = make_shared<v3::Broadcast>(const_zero, dense_shape);
    } else {
        auto default_value = node.get_input(3);
        broadcast = make_shared<v3::Broadcast>(default_value, dense_shape);
    }

    auto scatternd = make_shared<v3::ScatterNDUpdate>(broadcast, indices, values);
    set_node_name(node.get_name(), scatternd);
    return {scatternd};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
