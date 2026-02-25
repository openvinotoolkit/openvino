// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/iterator.hpp"

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/op/parameter.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_iterator_op(const ov::frontend::tensorflow::NodeContext& node) {
    default_op_checks(node, 0, {"Iterator", "IteratorV2", "OneShotIterator"});
    // retrieve all attributes
    auto container = node.get_attribute<string>("container");
    auto shared_name = node.get_attribute<string>("shared_name");
    auto output_types = node.get_attribute<vector<element::Type>>("output_types");
    auto output_shapes = node.get_attribute<vector<PartialShape>>("output_shapes");

    auto iterator = make_shared<Iterator>(shared_name, container, output_types, output_shapes, node.get_decoder());
    set_node_name(node.get_name(), iterator);
    return {iterator};
}

OutputVector translate_iterator_get_next_op(const NodeContext& node) {
    // Iterator operations are responsible for iterator creation
    // and it usually goes to IteratorGetNext so we have to handle
    // Iterator->IteratorGetNext sub-graph
    // From IteratorGetNext generates nultiple outputs which we should
    // automatically prune and create Parameter node for each of them
    default_op_checks(node, 1, {"IteratorGetNext"});
    auto node_name = node.get_name();
    auto output_types = node.get_attribute<vector<element::Type>>("output_types");
    auto output_shapes = node.get_attribute<vector<PartialShape>>("output_shapes");

    size_t output_num = output_types.size();
    TENSORFLOW_OP_VALIDATION(
        node,
        output_num == output_shapes.size(),
        "[TensorFlow Frontend] Incorrect input model: lenghts of output_types and output_shapes do not match.");

    OutputVector iterator_get_next_outputs;
    // perform auto-pruning: create Parameter nodes for each output of IteratorGetNext
    for (size_t output_ind = 0; output_ind < output_num; ++output_ind) {
        auto output_type = output_types[output_ind];
        auto output_shape = output_shapes[output_ind];
        auto parameter = make_shared<v0::Parameter>(output_type, output_shape);
        if (output_num == 1) {
            set_node_name(node_name, parameter);
        } else {
            set_node_name(node_name + ":" + to_string(output_ind), parameter);
        }
        iterator_get_next_outputs.push_back(parameter);
    }
    return iterator_get_next_outputs;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
