// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "hash_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/variable.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_lookup_table_find_op(const NodeContext& node) {
    default_op_checks(node, 3, {"LookupTableFind", "LookupTableFindV2"});
    auto table_handle = as_type_ptr<HashTable>(node.get_input_by_reference(0).get_node_shared_ptr());
    TENSORFLOW_OP_VALIDATION(
        node,
        table_handle,
        "[TensorFlow Frontend] internal error: LookupTableFind operation expects table_handle by the first input");
    TENSORFLOW_OP_VALIDATION(
        node,
        table_handle->is_initialized(),
        "[TensorFlow Frontend] internal error: LookupTableFind operation expects initialized table_handle");
    auto keys = node.get_input(1);
    auto default_value = node.get_input(2);

    auto key_type = table_handle->get_key_type();
    TENSORFLOW_OP_VALIDATION(
        node,
        key_type.is_integral_number(),
        "[TensorFlow Frontend] internal error: LookupTableFind is only supported for integer keys");

    auto all_keys = table_handle->get_values();
    auto all_values = table_handle->get_keys();

    // reshape both all values and keys to 1D tensor to work it further
    auto target_shape = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{-1});
    all_keys = make_shared<v1::Reshape>(all_keys, target_shape, false);
    all_values = make_shared<v1::Reshape>(all_values, target_shape, false);

    // update all values with default value and all keys
    auto default_value_shape = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{1});
    default_value = make_shared<v1::Reshape>(default_value, default_value_shape, false);
    all_values = make_shared<v0::Concat>(OutputVector{all_values, default_value}, 0);
    auto key_for_default_value = make_shared<v3::ShapeOf>(all_keys, element::i64)->output(0);
    key_for_default_value = make_shared<v0::Convert>(key_for_default_value, key_type);
    all_keys = make_shared<v0::Concat>(OutputVector{all_keys, key_for_default_value}, 0);

    // compute mask which keys are not valid and for which default value must be used
    auto unsqueeze_axis = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{-1});
    auto unsqueeze_keys = make_shared<v0::Unsqueeze>(keys, unsqueeze_axis);
    auto equal_mask = make_shared<v1::Equal>(all_keys, unsqueeze_keys)->output(0);
    equal_mask = make_shared<v1::ReduceLogicalOr>(equal_mask, unsqueeze_axis, false);

    // TODO: ... mapping is needed

    // replace invalid keys with key_for_default_value
    keys = make_shared<v1::Select>(equal_mask, keys, key_for_default_value);

    // at this point all keys are sorted and are from the range [0, n]
    // and keys are also mapped to this range
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{0});
    auto lookup_values = make_shared<v8::Gather>(all_values, keys, gather_axis);
    set_node_name(node.get_name(), lookup_values);

    return {lookup_values};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
