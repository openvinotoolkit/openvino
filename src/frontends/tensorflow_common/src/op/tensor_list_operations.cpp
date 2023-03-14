// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;
using namespace opset10;

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
    auto empty_constant = make_shared<Constant>(element_dtype, Shape{0, 0});
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
    Output<Node> num_elements = make_shared<ShapeOf>(input_handle, element::i32);
    auto zero_const = make_shared<Constant>(element::i32, Shape{1}, 0);
    auto one_const = make_shared<Constant>(element::i32, Shape{1}, 1);
    num_elements = make_shared<Slice>(num_elements, zero_const, one_const, one_const);

    // restore the real shape of tensor elements
    auto new_shape = make_shared<Concat>(OutputVector{num_elements, element_shape}, 0);
    auto reshape = make_shared<Reshape>(input_handle, new_shape, false);

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
    index = make_shared<Squeeze>(index);

    // gather tensor element by the required position
    auto gather_axis = make_shared<Constant>(element::i32, Shape{1}, 0);
    Output<Node> tensor_element = make_shared<Gather>(input_handle, index, gather_axis);
    tensor_element = make_shared<Convert>(tensor_element, element_dtype);

    set_node_name(node.get_name(), tensor_element.get_node_shared_ptr());
    return {tensor_element};
}

OutputVector translate_tensor_list_set_item_op(const NodeContext& node) {
    default_op_checks(node, 3, {"TensorListSetItem"});
    auto input_handle = node.get_input(0);
    auto index = node.get_input(1);
    auto item = node.get_input(2);

    // squeeze index tensor to have a scalar
    index = make_shared<Squeeze>(index);

    // flatten item to be inserted since
    // the tensor list saves elements in the flatten form
    auto new_item_shape = make_shared<Constant>(element::i32, Shape{1}, -1);
    item = make_shared<Reshape>(item, new_item_shape, false);
    auto item_shape = make_shared<ShapeOf>(item, element::i32);

    // reshape the tensor list to the shape [num_elements, -1]
    // that is because in the first iteration we have empty constant of a shape [0,0]
    auto minus_one = make_shared<Constant>(element::i32, Shape{1}, -1);
    auto new_input_handle_shape = make_shared<Concat>(OutputVector{minus_one, item_shape}, 0);
    input_handle = make_shared<Reshape>(input_handle, new_input_handle_shape, false);
    input_handle = make_shared<Convert>(input_handle, item.get_element_type());

    // compute the current length of the list
    Output<Node> list_length = make_shared<ShapeOf>(input_handle, element::i32);
    auto zero_const = make_shared<Constant>(element::i32, Shape{1}, 0);
    auto one_const = make_shared<Constant>(element::i32, Shape{1}, 1);
    list_length = make_shared<Slice>(list_length, zero_const, one_const, one_const);

    // compute a size of the dummy tensor that serves to fill holes in the list
    // if no tensor is inserted at this position
    auto one_const_scalar = make_shared<Constant>(element::i32, Shape{1}, 1);
    auto index_plus_one = make_shared<Add>(index, one_const_scalar);
    Output<Node> max_length = make_shared<Maximum>(list_length, index_plus_one);
    Output<Node> dummy_tensor_size = make_shared<Subtract>(max_length, list_length);

    // create dummy tensor and concatenate it
    auto zero_element = make_shared<Constant>(item.get_element_type(), Shape{}, 0);
    auto dummy_tensor_shape = make_shared<Concat>(OutputVector{dummy_tensor_size, item_shape}, 0);
    auto dummy_tensor = make_shared<Broadcast>(zero_element, dummy_tensor_shape);
    input_handle = make_shared<Concat>(OutputVector{input_handle, dummy_tensor}, 0);

    // update the resulted tensor using ScatterUpdate
    index = make_shared<Unsqueeze>(index, zero_const);
    item = make_shared<Unsqueeze>(item, zero_const);
    auto scatter_update = make_shared<ScatterUpdate>(input_handle, index, item, zero_const);

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
    auto new_tensor_shape = make_shared<Constant>(element::i32, Shape{1}, -1);
    tensor = make_shared<Reshape>(tensor, new_tensor_shape, false);
    auto tensor_shape = make_shared<ShapeOf>(tensor, element::i32);

    // reshape the tensor list to the shape [num_elements, -1]
    // that is because in the first iteration we have empty constant of a shape [0,0]
    Output<Node> num_elements = make_shared<ShapeOf>(input_handle, element::i32);
    auto zero_const = make_shared<Constant>(element::i32, Shape{1}, 0);
    auto one_const = make_shared<Constant>(element::i32, Shape{1}, 1);
    num_elements = make_shared<Slice>(num_elements, zero_const, one_const, one_const);
    auto new_input_handle_shape = make_shared<Concat>(OutputVector{num_elements, tensor_shape}, 0);
    input_handle = make_shared<Reshape>(input_handle, new_input_handle_shape, false);

    // unsqueeze tensor to be inserted into the list
    tensor = make_shared<Unsqueeze>(tensor, zero_const);

    // insert the tensor into the end
    auto updated_list = make_shared<Concat>(OutputVector{input_handle, tensor}, 0);

    set_node_name(node.get_name(), updated_list);
    return {updated_list};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
