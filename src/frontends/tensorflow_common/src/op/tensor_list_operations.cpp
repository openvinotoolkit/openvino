// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_tensor_list_reserve_op(const NodeContext& node) {
    default_op_checks(node, 2, {"TensorListReserve", "EmptyTensorList"});
    auto element_dtype = node.get_attribute<element::Type>("element_dtype");

    // always reserve an empty constant of rank equal to two
    // all tensor elements will be saved in the flatten form in the list
    // because we want to cover a case of dynamic rank tensor list
    // the real shape of the tensor elements will be restored by TensorListStack operations
    auto empty_constant = make_shared<v0::Constant>(element_dtype, Shape{0, 0});
    set_node_name(node.get_name(), empty_constant);
    return {empty_constant};
}

OutputVector translate_tensor_list_from_tensor_op(const NodeContext& node) {
    default_op_checks(node, 2, {"TensorListFromTensor"});
    auto tensor = node.get_input(0);

    // nothing to do, the input tensor will simulate the tensor list
    return {tensor};
}

OutputVector translate_tensor_list_stack_op(const NodeContext& node) {
    default_op_checks(node, 2, {"TensorListStack"});
    auto input_handle = node.get_input(0);
    auto element_shape = node.get_input(1);

    // compute number of tensor elements in the list
    Output<Node> num_elements = make_shared<v3::ShapeOf>(input_handle, element::i32);
    auto zero_const = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto one_const = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    num_elements = make_shared<v8::Slice>(num_elements, zero_const, one_const, one_const);

    // restore the real shape of tensor elements
    auto new_shape = make_shared<v0::Concat>(OutputVector{num_elements, element_shape}, 0);
    auto reshape = make_shared<v1::Reshape>(input_handle, new_shape, false);

    set_node_name(node.get_name(), reshape);
    return {reshape};
}

OutputVector translate_tensor_list_get_item_op(const NodeContext& node) {
    default_op_checks(node, 3, {"TensorListGetItem"});
    auto input_handle = node.get_input(0);
    auto index = node.get_input(1);
    auto element_shape = node.get_input(2);
    auto element_dtype = node.get_attribute<element::Type>("element_dtype");

    // squeeze index tensor to have a scalar
    index = make_shared<v0::Squeeze>(index);

    // gather tensor element by the required position
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    Output<Node> tensor_element = make_shared<v8::Gather>(input_handle, index, gather_axis);
    tensor_element = make_shared<v0::Convert>(tensor_element, element_dtype);

    set_node_name(node.get_name(), tensor_element.get_node_shared_ptr());
    return {tensor_element};
}

OutputVector translate_tensor_list_set_item_op(const NodeContext& node) {
    default_op_checks(node, 3, {"TensorListSetItem"});
    auto input_handle = node.get_input(0);
    auto index = node.get_input(1);
    auto item = node.get_input(2);

    // squeeze index tensor to have a scalar
    index = make_shared<v0::Squeeze>(index);

    // flatten item to be inserted since
    // the tensor list saves elements in the flatten form
    auto new_item_shape = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
    item = make_shared<v1::Reshape>(item, new_item_shape, false);
    auto item_shape = make_shared<v3::ShapeOf>(item, element::i32);

    // reshape the tensor list to the shape [num_elements, -1]
    // that is because in the first iteration we have empty constant of a shape [0,0]
    auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
    auto new_input_handle_shape = make_shared<v0::Concat>(OutputVector{minus_one, item_shape}, 0);
    input_handle = make_shared<v1::Reshape>(input_handle, new_input_handle_shape, false);
    input_handle = make_shared<v1::ConvertLike>(input_handle, item);

    // compute the current length of the list
    Output<Node> list_length = make_shared<v3::ShapeOf>(input_handle, element::i32);
    auto zero_const = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto one_const = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    list_length = make_shared<v8::Slice>(list_length, zero_const, one_const, one_const);

    // compute a size of the dummy tensor that serves to fill holes in the list
    // if no tensor is inserted at this position
    auto one_const_scalar = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    auto index_plus_one = make_shared<v1::Add>(index, one_const_scalar);
    Output<Node> max_length = make_shared<v1::Maximum>(list_length, index_plus_one);
    Output<Node> dummy_tensor_size = make_shared<v1::Subtract>(max_length, list_length);

    // create dummy tensor and concatenate it
    auto zero_element = create_same_type_const_scalar<int32_t>(item, 0);
    auto dummy_tensor_shape = make_shared<v0::Concat>(OutputVector{dummy_tensor_size, item_shape}, 0);
    auto dummy_tensor = make_shared<v3::Broadcast>(zero_element, dummy_tensor_shape);
    input_handle = make_shared<v0::Concat>(OutputVector{input_handle, dummy_tensor}, 0);

    // update the resulted tensor using ScatterUpdate
    index = make_shared<v0::Unsqueeze>(index, zero_const);
    item = make_shared<v0::Unsqueeze>(item, zero_const);
    auto scatter_update = make_shared<v3::ScatterUpdate>(input_handle, index, item, zero_const);

    set_node_name(node.get_name(), scatter_update);
    return {scatter_update};
}

OutputVector translate_tensor_list_push_back_op(const NodeContext& node) {
    default_op_checks(node, 2, {"TensorListPushBack"});
    auto input_handle = node.get_input(0);
    auto tensor = node.get_input(1);

    // flatten item to be inserted since
    // the tensor list saves elements in the flatten form
    // because we want to cover a case of dynamic rank tensor list
    // the real shape of the tensor elements will be restored by TensorListStack operations
    auto new_tensor_shape = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
    tensor = make_shared<v1::Reshape>(tensor, new_tensor_shape, false);
    auto tensor_shape = make_shared<v3::ShapeOf>(tensor, element::i32);

    // reshape the tensor list to the shape [num_elements, -1]
    // that is because in the first iteration we have empty constant of a shape [0,0]
    Output<Node> num_elements = make_shared<v3::ShapeOf>(input_handle, element::i32);
    auto zero_const = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto one_const = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    num_elements = make_shared<v8::Slice>(num_elements, zero_const, one_const, one_const);
    auto new_input_handle_shape = make_shared<v0::Concat>(OutputVector{num_elements, tensor_shape}, 0);
    input_handle = make_shared<v1::Reshape>(input_handle, new_input_handle_shape, false);

    // unsqueeze tensor to be inserted into the list
    tensor = make_shared<v0::Unsqueeze>(tensor, zero_const);

    // insert the tensor into the end
    auto updated_list = make_shared<v0::Concat>(OutputVector{input_handle, tensor}, 0);

    set_node_name(node.get_name(), updated_list);
    return {updated_list};
}

OutputVector translate_tensor_list_resize_op(const NodeContext& node) {
    default_op_checks(node, 2, {"TensorListResize"});
    auto input_handle = node.get_input(0);
    auto size = node.get_input(1);

    // create auxiliary constants
    auto zero_const = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto one_const = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    auto max_const = make_shared<v0::Constant>(element::i32, Shape{1}, numeric_limits<int32_t>::max());

    // compute the current length of the list and item shape
    auto tensor_list_shape = make_shared<v3::ShapeOf>(input_handle, element::i32);
    auto list_length = make_shared<v8::Slice>(tensor_list_shape, zero_const, one_const, one_const);
    auto item_shape = make_shared<v8::Slice>(tensor_list_shape, one_const, max_const, one_const);

    // compute a size of the dummy tensor to resize
    // and clip it by zero if it is negative
    Output<Node> dummy_tensor_size = make_shared<v1::Subtract>(size, list_length);
    dummy_tensor_size = make_shared<v1::Maximum>(dummy_tensor_size, zero_const);

    // create dummy tensor and concatenate it
    auto zero_const_same_type = create_same_type_const<float>(input_handle, vector<float>{0.0f}, Shape{});
    auto dummy_tensor_shape = make_shared<v0::Concat>(OutputVector{dummy_tensor_size, item_shape}, 0);
    auto dummy_tensor = make_shared<v3::Broadcast>(zero_const_same_type, dummy_tensor_shape);
    input_handle = make_shared<v0::Concat>(OutputVector{input_handle, dummy_tensor}, 0);

    // reshape size to have 1D tensor with one element
    auto new_size_shape = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    size = make_shared<v1::Reshape>(size, new_size_shape, false);

    // resize can also shrink the input tensor list
    input_handle = make_shared<v8::Slice>(input_handle, zero_const, size, one_const);

    set_node_name(node.get_name(), input_handle.get_node_shared_ptr());
    return {input_handle};
}

OutputVector translate_tensor_list_length_op(const NodeContext& node) {
    default_op_checks(node, 1, {"TensorListLength"});
    auto input_handle = node.get_input(0);

    // create auxiliary constants
    auto zero_const = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto one_const = make_shared<v0::Constant>(element::i32, Shape{1}, 1);

    // compute the current length of the list
    auto tensor_list_shape = make_shared<v3::ShapeOf>(input_handle, element::i32);
    auto list_length = make_shared<v8::Slice>(tensor_list_shape, zero_const, one_const, one_const);

    set_node_name(node.get_name(), list_length);
    return {list_length};
}

OutputVector translate_tensor_list_concat_v2_op(const NodeContext& node) {
    default_op_checks(node, 2, {"TensorListConcatV2"});
    auto input_handle = node.get_input(0);
    auto size = node.get_input(1);

    std::vector<int64_t> leading_dims;
    get_const_input(node, 2, &leading_dims);

    TENSORFLOW_OP_VALIDATION(node,
                             leading_dims.size() == 0,
                             "TensorListConcatV2 is not supported for non-empty leading_dims.");

    TENSORFLOW_OP_VALIDATION(node,
                             as_type_ptr<v0::Constant>(node.get_input(1).get_node_shared_ptr()),
                             "TensorListConcatV2 is not supported with non-constant shape input");

    std::vector<int64_t> list_elememt_shape;
    get_const_input(node, 1, &list_elememt_shape);

    list_elememt_shape[0] = list_elememt_shape[0] * input_handle.get_partial_shape()[0].get_max_length();
    auto out = make_shared<v1::Reshape>(
        input_handle,
        make_shared<v0::Constant>(element::i64, Shape{list_elememt_shape.size()}, list_elememt_shape),
        false);

    set_node_name(node.get_name(), out);

    return {out};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
