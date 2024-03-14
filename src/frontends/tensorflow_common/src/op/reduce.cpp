// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

template <typename T>
OutputVector translate_direct_reduce_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Any", "All", "EuclideanNorm", "Max", "Mean", "Min", "Prod", "Sum"}, true);
    auto input = node.get_input(0);
    auto axis = node.get_input(1);
    auto keep_dims = node.get_attribute<bool>("keep_dims", false);
    auto bool is_input_complex =
        input->get_data_type() == DataType::Complex64 || input->get_data_type() == DataType::Complex128;
    if (is_input_complex) {
        auto complex_type_mark = input->get_attribute<ComplexTypeMark>("ComplexTypeMark");
        auto output_shape = input->get_shape();
        output_shape.push_back(2);  // Adding auxillary dimension

        auto output_tensor = make_shared<Tensor>(DataType::Float32, output_shape);
        output_tensor->set_attribute("ComplexTypeMark", complex_type_mark);

        auto reduce_op = make_shared<T>(input, axis, keep_dims);
        set_node_name(node.get_name(), reduce_op);
        return {reduce_op, output_tensor};
    } else {
        auto reduce_op = make_shared<T>(input, axis, keep_dims);
        set_node_name(node.get_name(), reduce_op);
        return {reduce_op};
    }
}

template OutputVector translate_direct_reduce_op<v1::ReduceLogicalOr>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceLogicalAnd>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceMax>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceMean>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceMin>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceProd>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceSum>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v4::ReduceL2>(const NodeContext& node);
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
