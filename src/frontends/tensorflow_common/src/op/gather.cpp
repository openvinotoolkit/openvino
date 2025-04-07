// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather.hpp"

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
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
    default_op_checks(node, 2, {"Gather"}, true);
    auto params = node.get_input(0);
    auto axis = make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(params.get_node_shared_ptr());

    if (complex_type_mark) {
        auto complex_part_type = complex_type_mark->get_complex_part_type();
        params = complex_type_mark->get_data();
        auto indices = node.get_input(1);
        auto gather = make_shared<v8::Gather>(params, indices, axis, 0);
        set_node_name(node.get_name(), gather);
        // Return the Gather result directly without ComplexTypeMark
        return {gather};
    }

    return translate_basic_gather_op(node, axis, 0);
}

OutputVector translate_resource_gather_op(const NodeContext& node) {
    default_op_checks(node, 2, {"ResourceGather"});
    auto axis = make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    return translate_basic_gather_op(node, axis, batch_dims);
}

// Update the translate_gather_v2_op function
OutputVector translate_gather_v2_op(const NodeContext& node) {
    default_op_checks(node, 3, {"GatherV2"}, true);
    auto params = node.get_input(0);
    auto axis = node.get_input(2);
    auto batch_dims = node.get_attribute<int64_t>("batch_dims", 0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(params.get_node_shared_ptr());

    if (complex_type_mark) {
        auto complex_part_type = complex_type_mark->get_complex_part_type();
        params = complex_type_mark->get_data();
        auto indices = node.get_input(1);

        // Calculate the axis without subtracting 1 for complex tensors
        auto input_rank = params.get_partial_shape().rank().get_length();
        auto axis_val = axis.get_node_shared_ptr()->get_attribute<int64_t>("value")[0];
        if (axis_val < 0) {
            axis_val += input_rank;  // Use input_rank directly, not input_rank - 1
        }

        auto updated_axis = std::make_shared<v0::Constant>(element::i64, Shape{}, axis_val);
        auto gather = make_shared<v8::Gather>(params, indices, updated_axis, batch_dims);
        set_node_name(node.get_name(), gather);
        return {gather};  // Remove ComplexTypeMark from output
    }

    return translate_basic_gather_op(node, axis, batch_dims);
}

OutputVector translate_gather_nd_op(const NodeContext& node) {
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