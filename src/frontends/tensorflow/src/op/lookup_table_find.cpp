// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/hash_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/variable.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
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
    auto keys = node.get_input(1);
    auto default_value = node.get_input(2);

    auto key_type = table_handle->get_key_type();
    TENSORFLOW_OP_VALIDATION(
        node,
        key_type.is_integral_number(),
        "[TensorFlow Frontend] internal error: LookupTableFind is only supported for integer keys");

    auto all_keys = table_handle->get_keys();
    auto all_values = table_handle->get_values();

    // reshape both all values and keys to 1D tensor to work it further
    auto target_shape = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{-1});
    all_keys = make_shared<v1::Reshape>(all_keys, target_shape, false);
    all_values = make_shared<v1::Reshape>(all_values, target_shape, false);

    // update all values with default value and all keys
    auto default_value_shape = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{1});
    default_value = make_shared<v1::Reshape>(default_value, default_value_shape, false);
    all_values = make_shared<v0::Concat>(OutputVector{all_values, default_value}, 0);
    auto num_keys = make_shared<v3::ShapeOf>(all_keys, element::i64)->output(0);
    auto scalar_shape = make_shared<v0::Constant>(element::i32, Shape{0}, vector<int32_t>{});
    num_keys = make_shared<v1::Reshape>(num_keys, scalar_shape, false);
    num_keys = make_shared<v0::Convert>(num_keys, key_type);

    // compute mask which keys are not valid and for which default value must be used
    auto unsqueeze_axis = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{-1});
    auto unsqueeze_keys = make_shared<v0::Unsqueeze>(keys, unsqueeze_axis);
    auto equal_mask = make_shared<v1::Equal>(all_keys, unsqueeze_keys)->output(0);
    auto reduce_equal_mask = make_shared<v1::ReduceLogicalOr>(equal_mask, unsqueeze_axis, false);

    // map keys to new keys from range [0, n], n index will be for out-of-range keys
    // 1. generate mask-01 of shape [keys_shape, len(all_keys)],
    // where 0 - not found key, 1 - found key
    auto const_zero = make_shared<v0::Constant>(key_type, Shape{}, 0);
    auto const_one = make_shared<v0::Constant>(key_type, Shape{}, 1);
    auto mask01 = make_shared<v1::Select>(equal_mask, const_one, const_zero);
    // 2. generate a range [0, n-1] that will be multiplied to mask for computation of new keys
    auto new_all_keys = make_shared<v4::Range>(const_zero, num_keys, const_one, key_type);
    // 3. compute new keys
    auto reduce_axis = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{-1});
    auto new_keys = make_shared<v1::Multiply>(mask01, new_all_keys)->output(0);
    new_keys = make_shared<v1::ReduceMax>(new_keys, reduce_axis, false);

    // replace invalid keys with key_for_default_value
    new_keys = make_shared<v1::Select>(reduce_equal_mask, new_keys, num_keys);

    // at this point all keys are sorted and are from the range [0, n]
    // and keys are also mapped to this range
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{0});
    auto lookup_values = make_shared<v8::Gather>(all_values, new_keys, gather_axis);
    set_node_name(node.get_name(), lookup_values);

    return {lookup_values};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
