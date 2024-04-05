// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/enter.hpp"
#include "helper_ops/tensor_array.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

namespace {
// the function creates the constant imitating initial tensor array container
Output<Node> create_initial_tensor_array_constant(int64_t tensor_element_rank,
                                                  const element::Type& element_type,
                                                  Output<Node> size,
                                                  const string& node_name) {
    // adjust size to have it of shape [1] for further concatenation with element shape
    auto new_size_shape = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    size = make_shared<v1::Reshape>(size, new_size_shape, false);

    // create a vector of size element_shape.rank() with ones
    // and compute a shape of initial tensor array [size, 1, ..., 1]
    vector<int32_t> ones(tensor_element_rank, 1);
    auto ones_const = make_shared<v0::Constant>(element::i32, Shape{ones.size()}, ones);
    auto target_shape = make_shared<v0::Concat>(OutputVector{size, ones_const}, 0);

    // create initial tensor array
    auto scalar_value = make_shared<v0::Constant>(element_type, Shape{}, vector<int32_t>{0});
    auto initial_tensor_array = make_shared<v3::Broadcast>(scalar_value, target_shape);

    return initial_tensor_array->output(0);
}
}  // namespace

OutputVector translate_tensor_array_v3_op(const NodeContext& node) {
    // TensorArrayV3 has just one input:
    // 0) size to initialize a size of tensor array
    default_op_checks(node, 1, {"TensorArrayV3"});
    auto dtype = node.get_attribute<element::Type>("dtype");
    auto size = node.get_input(0);
    auto element_shape = node.get_attribute<PartialShape>("element_shape", ov::PartialShape::dynamic());
    bool dynamic_size = node.get_attribute<bool>("dynamic_size", false);
    int64_t element_rank = element_shape.rank().is_static() ? element_shape.rank().get_length() : -1;

    if (element_rank != -1 && !dynamic_size) {
        auto node_name = node.get_name();
        auto new_output1 =
            create_initial_tensor_array_constant(element_shape.rank().get_length(), dtype, size, node.get_name());
        new_output1.set_names({node_name + ":0"});
        auto new_output2 =
            create_initial_tensor_array_constant(element_shape.rank().get_length(), dtype, size, node.get_name());
        new_output2.set_names({node_name + ":1"});
        return OutputVector{new_output1, new_output2};
    }

    // dynamic case when it is unable retrieve element rank from the attribute or container size is dynamic
    auto tensor_array_v3 = make_shared<TensorArrayV3>(size, dtype, element_rank, dynamic_size, node.get_decoder());
    set_node_name(node.get_name(), tensor_array_v3);

    return tensor_array_v3->outputs();
}

OutputVector translate_tensor_array_scatter_v3_op(const NodeContext& node) {
    // TensorArrayScatterV3 has four inputs:
    // 0) handle, a Tensor of type resource. The handle to a TensorArray.
    // 1) indices, a Tensor of type int32. The locations at which to write the tensor elements.
    // 2) value, a Tensor. The concatenated tensor to write to the TensorArray
    // 3) flow_in A Tensor of type float32. A float scalar that enforces proper chaining of operations.
    // The operation has one output:
    // 0) flow_out indicates that operation is complete and handle resource is updated
    default_op_checks(node, 4, {"TensorArrayScatterV3"});
    auto indices = node.get_input(1);
    auto value = node.get_input(2);
    // flow_in is used for transferring input tensor array
    auto tensor_array = node.get_input(3);

    // check if producer of tensor_array is TensorArrayV3, internal operation, still
    // if yes, try to replace it with constant container
    if (as_type_ptr<TensorArrayV3>(tensor_array.get_node_shared_ptr()) &&
        value.get_partial_shape().rank().is_static()) {
        // set tensor element rank that gets known from TensorArrayScatterV3 operation
        auto tensor_array_v3 = as_type_ptr<TensorArrayV3>(tensor_array.get_node_shared_ptr());
        TENSORFLOW_OP_VALIDATION(
            node,
            value.get_partial_shape().rank().get_length() > 0,
            "[TensorFlow Frontend] internal error or inconsistent model: value to TensorArrayScatterV3 is a scalar");
        int64_t tensor_element_rank = value.get_partial_shape().rank().get_length() - 1;
        tensor_array_v3->set_element_rank(tensor_element_rank);
    }

    // compute element shape (shape of a tensor in the tensor array) using value
    auto element_shape = make_shared<v3::ShapeOf>(value, element::i32)->output(0);
    auto one_const = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    auto max_const = make_shared<v0::Constant>(element::i32, Shape{1}, numeric_limits<int32_t>::max());
    element_shape = make_shared<v8::Slice>(element_shape, one_const, max_const, one_const);

    // compute size of tensor array
    auto tensor_array_size = make_shared<v3::ShapeOf>(tensor_array, element::i32)->output(0);
    auto zero_const = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    tensor_array_size = make_shared<v8::Gather>(tensor_array_size, zero_const, zero_const);

    // compute the new shape for tensor array where new tensors will be inserted
    auto new_shape = make_shared<v0::Concat>(OutputVector{tensor_array_size, element_shape}, 0);
    tensor_array = make_shared<v3::Broadcast>(tensor_array, new_shape);

    // adjust indices for ScatterNDUpdate to have a shape [N, 1] where N is a number of indices
    indices = make_shared<v0::Unsqueeze>(indices, one_const);

    // compute updated tensor array using ScatterNDUpdate
    // value should be of a shape [N, <elem_shape>]
    auto updated_tensor_array = make_shared<v3::ScatterNDUpdate>(tensor_array, indices, value);
    set_node_name(node.get_name(), updated_tensor_array);

    // TensorArrayScatterV3 has just one output flow_out
    // that is used for transferring updated tensor array
    return {updated_tensor_array};
}

OutputVector translate_tensor_array_read_v3_op(const NodeContext& node) {
    // TensorArrayReadV3 read an element from the TensorArray into the output
    // and it has three inputs:
    // 0) handle, a Tensor of type resource. The handle to a TensorArray.
    // 1) index, a Tensor of type int32. The location from which to read the value
    // 2) flow_in A Tensor of type float32. A float scalar that enforces proper chaining of operations.
    // The operation has one output
    // 0) read value from tensor array
    default_op_checks(node, 3, {"TensorArrayReadV3"});
    auto index = node.get_input(1);
    // flow_in is used for transferring input tensor array
    auto tensor_array = node.get_input(2);
    auto dtype = node.get_attribute<element::Type>("dtype");

    // adjust the index to a scalar for using Gather operation
    auto new_shape = make_shared<v0::Constant>(element::i32, Shape{0}, vector<int32_t>{});
    index = make_shared<v1::Reshape>(index, new_shape, false);

    // gather tensor element by the required position
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    Output<Node> tensor_element = make_shared<v8::Gather>(tensor_array, index, gather_axis);
    tensor_element = make_shared<v0::Convert>(tensor_element, dtype);

    set_node_name(node.get_name(), tensor_element.get_node_shared_ptr());
    return {tensor_element};
}

OutputVector translate_tensor_array_close_v3_op(const NodeContext& node) {
    // TensorArrayCloseV3 deletes the TensorArray from its resource container
    // it outputs nothing
    default_op_checks(node, 1, {"TensorArrayCloseV3"});
    return {};
}

OutputVector translate_tensor_array_size_v3_op(const NodeContext& node) {
    // TensorArraySizeV3 gets the current size of the TensorArray
    // it outputs int32 scalar equal to a size of the tensor array
    default_op_checks(node, 2, {"TensorArraySizeV3"});
    // skip the handle by the first input
    auto tensor_array = node.get_input(1);

    auto size = make_shared<v3::ShapeOf>(tensor_array, element::i32)->output(0);
    auto zero_const = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    size = make_shared<v8::Gather>(size, zero_const, zero_const);

    // size must be scalar
    auto scalar_shape = make_shared<v0::Constant>(element::i32, Shape{0}, vector<int32_t>{});
    size = make_shared<v1::Reshape>(size, scalar_shape, false);

    set_node_name(node.get_name(), size.get_node_shared_ptr());
    return {size};
}

OutputVector translate_tensor_array_gather_v3_op(const NodeContext& node) {
    // TensorArrayGatherV3 gathers specific elements from the TensorArray into output
    // and it has three inputs:
    // 0) handle, a Tensor of type resource. The handle to a TensorArray.
    // 1) indices, a Tensor of type int32. The location from which to read tensor elements
    // 2) flow_in A Tensor of type float32. A float scalar that enforces proper chaining of operations.
    // The operation has one output
    // 0) value with read tensor elements
    // it outputs int32 scalar equal to a size of the tensor array
    default_op_checks(node, 3, {"TensorArrayGatherV3"});
    // skip the handle by the first input
    auto indices = node.get_input(1);
    // flow_in serves for transferring tensor array
    // handle input is ignored
    auto tensor_array = node.get_input(2);
    auto dtype = node.get_attribute<element::Type>("dtype");
    auto element_shape = node.get_attribute<PartialShape>("element_shape", PartialShape::dynamic());

    // gather tensor element by the required position
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    Output<Node> tensor_element = make_shared<v8::Gather>(tensor_array, indices, gather_axis);
    tensor_element = make_shared<v0::Convert>(tensor_element, dtype);

    // concretize tensor_element shape if this is specified
    if (tensor_element.get_partial_shape().rank().is_dynamic() && element_shape.is_static()) {
        auto element_shape_value = element_shape.get_shape();
        auto element_shape_const =
            make_shared<v0::Constant>(element::i32, Shape{element_shape_value.size()}, element_shape_value);
        auto size = make_shared<v3::ShapeOf>(tensor_array, element::i32)->output(0);
        auto zero_const = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
        size = make_shared<v8::Gather>(size, zero_const, zero_const);
        auto new_shape = make_shared<v0::Concat>(OutputVector{size, element_shape_const}, 0);
        tensor_element = make_shared<v1::Reshape>(tensor_element, new_shape, false);
    }

    set_node_name(node.get_name(), tensor_element.get_node_shared_ptr());
    return {tensor_element};
}

OutputVector translate_tensor_array_concat_v3_op(const NodeContext& node) {
    // TensorArrayConcatV3 Concat the elements from the TensorArray into value
    // and it has two inputs:
    // 0) handle, a Tensor of type resource. The handle to a TensorArray.
    // 1) flow_in A Tensor of type float32. A float scalar that enforces proper chaining of operations.
    // The operation has one output
    // 0) concatenated value by the first dimension
    default_op_checks(node, 2, {"TensorArrayConcatV3"});
    // flow_in serves for transferring tensor array
    // handle input is ignored
    auto tensor_array = node.get_input(1);
    auto dtype = node.get_attribute<element::Type>("dtype");

    // since tensor array saves tensor elements in the concatenated form by the first dimension
    // and for this operation they should be concatenated by the first dimension of the tensor element
    // it needs to combine the first two dimensions
    // tensor array is of shape [k, n0, n1, ..., nd]
    // 1. compute element shape excluding the first dimension
    auto zero_const = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto one_const = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    auto two_const = make_shared<v0::Constant>(element::i32, Shape{1}, 2);
    auto max_const = make_shared<v0::Constant>(element::i32, Shape{1}, numeric_limits<int32_t>::max());
    auto tensor_array_shape = make_shared<v3::ShapeOf>(tensor_array, element::i64);
    auto element_shape_no_two_dims = make_shared<v8::Slice>(tensor_array_shape, two_const, max_const, one_const);
    // 2. compute the first and second dimensions k and n0
    auto k = make_shared<v8::Gather>(tensor_array_shape, zero_const, zero_const);
    auto n0 = make_shared<v8::Gather>(tensor_array_shape, one_const, zero_const);
    auto k_by_n0 = make_shared<v1::Multiply>(k, n0);
    // 3. compute the first output containing concatenated tensor elements
    // it folds the first and second dimensions
    auto new_shape = make_shared<v0::Concat>(OutputVector{k_by_n0, element_shape_no_two_dims}, 0);
    auto concatenated_array = make_shared<v1::Reshape>(tensor_array, new_shape, false)->output(0);
    concatenated_array = make_shared<v0::Convert>(concatenated_array, dtype);
    concatenated_array.set_names({node.get_name() + ":0"});
    // 4. compute the second output with length of each tensor element for the concatenation
    auto lengths = make_shared<v3::Broadcast>(n0, k)->output(0);
    lengths.set_names({node.get_name() + ":1"});

    return {concatenated_array, lengths};
}

OutputVector translate_tensor_array_write_v3_op(const NodeContext& node) {
    // TensorArrayWriteV3 pushes an element onto the tensor_array.
    // and it has four inputs
    // 0) handle, a Tensor of type resource. The handle to a TensorArray.
    // 1) index, a Tensor of type int32. The location where to write tensor element
    // 2) value, a Tensor. The tensor to write at the specified location
    // 3) flow_in A Tensor of type float32. A float scalar that enforces proper chaining of operations.
    // The operation has one output
    // 0) read value from tensor array
    default_op_checks(node, 4, {"TensorArrayWriteV3"});
    auto handle = node.get_input(0);
    auto index = node.get_input(1);
    auto value = node.get_input(2);
    // flow_in is used for transferring input tensor array
    // tensor array has a rank equal to 1 + rank(element of tensor array)
    // if it just initialized, its shape is equal to [tensor_array_size, 1, ..., 1]
    // otherwise, it is equal to [tensor_array_size, <element shape>]
    auto tensor_array = node.get_input(3);
    bool dynamic_size = true;

    // reshape index to have it of [1] shape
    auto new_index_shape = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    index = make_shared<v1::Reshape>(index, new_index_shape, false);

    if (auto enter = as_type_ptr<Enter>(handle.get_node_shared_ptr())) {
        if (as_type_ptr<TensorArrayV3>(enter->input_value(0).get_node_shared_ptr()) &&
            value.get_partial_shape().rank().is_static()) {
            // set tensor element rank that gets known from TensorArrayWriteV3 operation
            auto tensor_array_v3 = as_type_ptr<TensorArrayV3>(enter->input_value(0).get_node_shared_ptr());
            int64_t tensor_element_rank = value.get_partial_shape().rank().get_length();
            tensor_array_v3->set_element_rank(tensor_element_rank);
            dynamic_size = tensor_array_v3->get_dynamic_size();
        }
    }

    // compute element shape in the input tensor array
    auto tensor_array_shape = make_shared<v3::ShapeOf>(tensor_array, element::i32);

    // compute the current size of tensor array
    auto zero_const = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto tensor_array_size = make_shared<v8::Gather>(tensor_array_shape, zero_const, zero_const);

    // adjust tensor array to have the correct shape [size, <real element shape>] before value insertion
    auto element_shape = make_shared<v3::ShapeOf>(value, element::i32);
    auto new_tensor_array_shape = make_shared<v0::Concat>(OutputVector{tensor_array_size, element_shape}, 0);
    tensor_array = make_shared<v3::Broadcast>(tensor_array, new_tensor_array_shape);

    if (dynamic_size) {
        // it requires to adjust a container size
        auto const_one = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
        auto index_plus_one = make_shared<v1::Add>(index, const_one);
        auto max_size = make_shared<v1::Maximum>(tensor_array_size, index_plus_one);

        auto dummy_size = make_shared<v1::Subtract>(max_size, tensor_array_size);
        auto dummy_tensor_shape = make_shared<v0::Concat>(OutputVector{dummy_size, element_shape}, 0);

        // create dummy tensor and concatenate it
        auto zero_element = create_same_type_const_scalar<int32_t>(value, 0);
        auto dummy_tensor = make_shared<v3::Broadcast>(zero_element, dummy_tensor_shape);
        tensor_array = make_shared<v0::Concat>(OutputVector{tensor_array, dummy_tensor}, 0);
    }

    // update the resulted tensor using ScatterUpdate
    value = make_shared<v0::Unsqueeze>(value, zero_const);
    auto scatter_update = make_shared<v3::ScatterUpdate>(tensor_array, index, value, zero_const);

    set_node_name(node.get_name(), scatter_update);
    // use flow_out for transferring updated tensor array
    return {scatter_update};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
