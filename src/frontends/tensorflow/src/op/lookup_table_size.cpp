// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/hash_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/variable.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_lookup_table_size_op(const NodeContext& node) {
    default_op_checks(node, 1, {"LookupTableSize", "LookupTableSizeV2"});
    auto table_handle = as_type_ptr<HashTable>(node.get_input_by_reference(0).get_node_shared_ptr());
    TENSORFLOW_OP_VALIDATION(
        node,
        table_handle,
        "[TensorFlow Frontend] internal error: LookupTableSize operation expects table_handle by the first input");

    auto all_keys = table_handle->get_keys();

    // reshape all keys to 1D tensor to work it further
    auto target_shape = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
    all_keys = make_shared<v1::Reshape>(all_keys, target_shape, false);

    // compute size of records in HashTable
    // table size must be a scalar
    ov::Output<ov::Node> table_size = make_shared<v3::ShapeOf>(all_keys, element::i64);
    auto squeeze_axis = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    table_size = make_shared<v0::Squeeze>(table_size, squeeze_axis);
    set_node_name(node.get_name(), table_size.get_node_shared_ptr());

    return {table_size};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
