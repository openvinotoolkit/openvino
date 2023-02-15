// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/fifo_queue.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset10;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_queue_dequeue_op(const NodeContext& node) {
    // QueueDequeue operation generates multiple outputs
    // which we prune and create Parameter node for
    vector<string> supported_ops = {"QueueDequeue",
                                    "QueueDequeueV2",
                                    "QueueDequeueUpTo",
                                    "QueueDequeueUpToV2",
                                    "QueueDequeueMany"};
    default_op_checks(node, 2, supported_ops);
    auto node_name = node.get_name();
    auto handle = node.get_input(0);
    auto n = node.get_input(1);

    // compute batch dimension for outputs
    // this is a number of batch objects emitted from QueueDequeue
    Dimension batch_dim = Dimension::dynamic();
    if (auto n_const = get_constant_from_source(n)) {
        auto n_value = n_const->cast_vector<int64_t>();
        if (n_value.size() > 0 && n_value[0] > 0) {
            batch_dim = n_value[0];
        }
    }

    // all data about output shapes and types are saved in FIFOQueue operation
    auto fifo_queue = as_type_ptr<FIFOQueue>(handle.get_node_shared_ptr());
    TENSORFLOW_OP_VALIDATION(
        node,
        fifo_queue,
        "[TensorFlow Frontend] Internal error: only FIFOQueue is supported as a producer for QueueDequeue operation.");
    auto output_shapes = fifo_queue->get_component_shapes();
    auto output_types = fifo_queue->get_component_types();
    size_t output_num = output_shapes.size();
    TENSORFLOW_OP_VALIDATION(
        node,
        output_num == output_types.size(),
        "[TensorFlow Frontend] Incorrect input model: lenghts of output_types and output_shapes do not match.");

    OutputVector queue_dequeue_outputs;
    // perform auto-pruning: create Parameter nodes for each output of QueueDequeue
    for (size_t output_ind = 0; output_ind < output_num; ++output_ind) {
        auto output_shape = output_shapes[output_ind];
        auto output_type = output_types[output_ind];
        output_shape.insert(output_shape.begin(), batch_dim);
        auto parameter = make_shared<Parameter>(output_type, output_shape);
        if (output_num == 1) {
            set_node_name(node_name, parameter);
        } else {
            set_node_name(node_name + ":" + to_string(output_ind), parameter);
        }
        queue_dequeue_outputs.push_back(parameter);
    }
    return queue_dequeue_outputs;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
