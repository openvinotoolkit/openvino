// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include "common_op_table.hpp"
#include "helper_ops/tensor_list_ops.hpp"
#include "openvino/core/validation_util.hpp"
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

namespace {
Output<Node> create_initial_tensor_list(const NodeContext& node,
                                        const Output<Node>& num_elements,
                                        const Output<Node>& element_shape,
                                        const element::Type& element_dtype) {
    // handle tensor elements of dynamic rank
    if (element_shape.get_partial_shape().is_dynamic() || element_shape.get_shape().size() != 1 ||
        element_dtype.is_dynamic()) {
        auto tensor_list = make_shared<TensorList>(num_elements, ov::Rank::dynamic(), element_dtype);
        return {tensor_list};
    }

    TENSORFLOW_OP_VALIDATION(node,
                             element_shape.get_partial_shape().is_static(),
                             "[TensorFlow Frontend] internal error: element_shape must be of static shape");

    TENSORFLOW_OP_VALIDATION(node,
                             element_shape.get_shape().size() == 1,
                             "[TensorFlow Frontend] inconsistent model: element_shape is not 1D vector");

    // create initial shape of elements
    size_t element_rank = static_cast<size_t>(element_shape.get_shape()[0]);
    auto initial_element_shape = make_shared<v0::Constant>(element::i32, Shape{element_rank}, 1);

    auto initial_tensor_list_shape = make_shared<v0::Concat>(OutputVector{num_elements, initial_element_shape}, 0);
    auto one_element = make_shared<v0::Constant>(element_dtype, Shape{}, 0);

    // create initial container of tensors with zeros and a shape equal to [num_elements, 1, ..., 1]
    auto initial_tensor_list = make_shared<v1::Broadcast>(one_element, initial_tensor_list_shape);

    return initial_tensor_list;
}
}  // namespace

OutputVector translate_tensor_list_reserve_op(const NodeContext& node) {
    default_op_checks(node, 2, {"TensorListReserve"});
    auto element_shape = node.get_input(0);
    // num_elements cannot be negative and define a number of elements in tensor list
    // trying setting element by out-of-bound index leads to failure
    auto num_elements = node.get_input(1);
    auto element_dtype = node.get_attribute<element::Type>("element_dtype");

    // provide num_elements of a shape [1]
    if (num_elements.get_partial_shape() != ov::PartialShape{1}) {
        auto new_num_elements_shape = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
        num_elements = make_shared<v1::Reshape>(num_elements, new_num_elements_shape, false);
    }

    auto initial_tensor_list = create_initial_tensor_list(node, num_elements, element_shape, element_dtype);

    set_node_name(node.get_name(), initial_tensor_list.get_node_shared_ptr());
    return {initial_tensor_list};
}

OutputVector translate_empty_tensor_list_op(const NodeContext& node) {
    default_op_checks(node, 2, {"EmptyTensorList"});
    auto element_shape = node.get_input(0);
    auto element_dtype = node.get_attribute<element::Type>("element_dtype");

    // a number of elements must be zero
    auto num_elements = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto initial_tensor_list = create_initial_tensor_list(node, num_elements, element_shape, element_dtype);

    set_node_name(node.get_name(), initial_tensor_list.get_node_shared_ptr());
    return {initial_tensor_list};
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

    // nothing to do, the input tensor will simulate the tensor list
    return {input_handle};
}

OutputVector translate_tensor_list_get_item_op(const NodeContext& node) {
    default_op_checks(node, 3, {"TensorListGetItem"});
    auto input_handle = node.get_input(0);
    auto index = node.get_input(1);
    auto element_shape = node.get_input(2);
    auto element_dtype = node.get_attribute<element::Type>("element_dtype");

    auto tensor_list_get_item = make_shared<TensorListGetItem>(input_handle, index, element_shape, element_dtype);
    set_node_name(node.get_name(), tensor_list_get_item);
    return {tensor_list_get_item};
}

OutputVector translate_tensor_list_set_item_op(const NodeContext& node) {
    default_op_checks(node, 3, {"TensorListSetItem"});
    auto input_handle = node.get_input(0);
    auto index = node.get_input(1);
    auto item = node.get_input(2);
    auto tensor_list_set_item = make_shared<TensorListSetItem>(input_handle, index, item);

    set_node_name(node.get_name(), tensor_list_set_item);
    return {tensor_list_set_item};
}

OutputVector translate_tensor_list_push_back_op(const NodeContext& node) {
    default_op_checks(node, 2, {"TensorListPushBack"});
    auto input_handle = node.get_input(0);
    auto tensor = node.get_input(1);

    auto tensor_list_push_back = make_shared<TensorListPushBack>(input_handle, tensor);

    set_node_name(node.get_name(), tensor_list_push_back);
    return {tensor_list_push_back};
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

    // output of TensorListLength must be a scalar
    // after Slice operation it is a 1D tensor with one element
    auto scalar_shape = make_shared<v0::Constant>(element::i32, Shape{0}, std::vector<int32_t>{});
    auto list_length_scalar = make_shared<v1::Reshape>(list_length, scalar_shape, false);

    set_node_name(node.get_name(), list_length_scalar);
    return {list_length_scalar};
}

OutputVector translate_tensor_list_concat_v2_op(const NodeContext& node) {
    // input tensor list (input_handle) is represented by a tensor
    // that is a result of concatenation of tensor list elements (tensors)
    // along unsqueezed zero dimension
    default_op_checks(node, 3, {"TensorListConcatV2"});
    auto input_handle = node.get_input(0);

    std::vector<int64_t> leading_dims;
    get_const_input(node, 2, &leading_dims);

    TENSORFLOW_OP_VALIDATION(node,
                             leading_dims.size() == 0,
                             "TensorListConcatV2 is not supported for non-empty leading_dims.");

    // tensor list shape has at least two dimensions:
    // 0-th dimension is auxiliary for elements concatenation
    // 1-st dimension is by definition of TensorListConcatV2 operation
    // along which elements will be concatenated
    // insert auxiliary 2-nd dimension to avoid Slice to operate on out-of-bound start value
    auto first_dim = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    auto two_dim = make_shared<v0::Constant>(element::i32, Shape{1}, 2);
    input_handle = make_shared<v0::Unsqueeze>(input_handle, two_dim);
    auto tensor_list_shape = make_shared<v3::ShapeOf>(input_handle, element::i64);

    // compute new_shape after TensorListConcatV2
    auto const_one = make_shared<v0::Constant>(ov::element::i32, Shape{1}, 1);
    auto stop = make_shared<v0::Constant>(ov::element::i32, Shape{1}, numeric_limits<int>::max());
    ov::Output<ov::Node> new_shape = make_shared<v8::Slice>(tensor_list_shape, two_dim, stop, const_one);
    auto const_minus_one = make_shared<v0::Constant>(ov::element::i64, Shape{1}, -1);
    new_shape = make_shared<v0::Concat>(ov::OutputVector{const_minus_one, new_shape}, 0);

    ov::Output<ov::Node> out = make_shared<v1::Reshape>(input_handle, new_shape, false);

    // do no forget about auxiliary dimension that became the first dimension
    out = make_shared<v0::Squeeze>(out, first_dim);

    set_node_name(node.get_name(), out.get_node_shared_ptr());
    return {std::move(out)};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
