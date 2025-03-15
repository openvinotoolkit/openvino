// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather.hpp"

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_basic_gather_op(const NodeContext& node, const ov::Output<ov::Node>& axis, int64_t batch_dims) {
    auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() >= 2, op_type + " must have at least two inputs.");
    auto params = node.get_input(0);
    auto indices = node.get_input(1);
    auto gather = make_shared<v8::Gather>(params, indices, axis, batch_dims);
    set_node_name(node.get_name(), gather);
    return {gather};
}

OutputVector translate_gather_op(const NodeContext& node) {
    // Gather has two inputs: data and indices
    // axis by which data is sliced is always equal to 0, batch_dims is always equal to 0
    default_op_checks(node, 2, {"Gather"}, true);
    auto params = node.get_input(0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(params.get_node_shared_ptr());
    auto axis = make_shared<v0::Constant>(element::i64, Shape{}, 0);

    if (complex_type_mark) {
        params = complex_type_mark->input_value(0);
        auto indices = node.get_input(1);
        auto gather = make_shared<v8::Gather>(params, indices, axis, 0);
        set_node_name(node.get_name(), gather);
        auto complex_reshape = make_shared<ComplexTypeMark>(gather, complex_type_mark->get_complex_part_type());
        return {complex_reshape};
    }

    return translate_basic_gather_op(node, axis, 0);
}

OutputVector translate_resource_gather_op(const NodeContext& node) {
    // ResourceGather has two inputs: data and indices
    // axis by which data is sliced is always equal to 0, batch_dims is an attribute and can vary
    default_op_checks(node, 2, {"ResourceGather"});
    auto axis = make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    return translate_basic_gather_op(node, axis, batch_dims);
}

OutputVector translate_gather_v2_op(const NodeContext& node) {
    // GatherV2 has three inputs: data, indices, and axis by which data is sliced
    // batch_dims is an attribute and can vary
    default_op_checks(node, 3, {"GatherV2"}, true);
    auto params = node.get_input(0);
    auto indices = node.get_input(1);
    auto axis = node.get_input(2);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(params.get_node_shared_ptr());

    if (complex_type_mark) {
        params = complex_type_mark->input_value(0);
        // If the axis is negative, adjust it
        auto zero = make_shared<v0::Constant>(ov::element::i32, Shape{}, 0);
        auto one = make_shared<v0::Constant>(ov::element::i32, Shape{}, 1);
        auto condition = make_shared<v1::Less>(axis, zero);
        auto updated_axis = make_shared<v1::Subtract>(axis, one);

        // create Select operation to choose between original axis and updated axis
        auto selected_axis = make_shared<v1::Select>(condition, updated_axis, axis);

        // Update batch_dims if negative
        auto updated_batch_dims = (batch_dims < 0) ? batch_dims - 1 : batch_dims;

        /// Create the Gather operation
        auto gather = make_shared<v8::Gather>(params, indices, selected_axis, updated_batch_dims);

        // Set the node's name and apply complex type marking if needed
        set_node_name(node.get_name(), gather);
        auto complex_gather = make_shared<ComplexTypeMark>(gather, complex_type_mark->get_complex_part_type());
        return {complex_gather};
    }

    return translate_basic_gather_op(node, axis, batch_dims);
}

OutputVector translate_gather_nd_op(const NodeContext& node) {
    // GatherND has two inputs: data and indices
    // batch_dims is always equal to 0
    default_op_checks(node, 2, {"GatherNd", "GATHER_ND"});
    auto input = node.get_input(0);
    auto input_indices = node.get_input(1);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    auto gather_nd = make_shared<v8::GatherND>(input, input_indices, batch_dims);
    set_node_name(node.get_name(), gather_nd);
    return {gather_nd};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
