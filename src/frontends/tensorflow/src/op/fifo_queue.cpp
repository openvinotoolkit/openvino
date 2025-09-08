// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/fifo_queue.hpp"

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_fifo_queue_op(const ov::frontend::tensorflow::NodeContext& node) {
    default_op_checks(node, 0, {"FIFOQueue", "FIFOQueueV2"});
    // retrieve all attributes
    auto component_types = node.get_attribute<vector<element::Type>>("component_types");
    auto shapes = node.get_attribute<vector<PartialShape>>("shapes");
    auto capacity = node.get_attribute<int64_t>("capacity", -1);
    auto container = node.get_attribute<string>("container", "");
    auto shared_name = node.get_attribute<string>("shared_name", "");

    auto fifo_queue =
        make_shared<FIFOQueue>(component_types, shapes, capacity, container, shared_name, node.get_decoder());
    set_node_name(node.get_name(), fifo_queue);
    return {fifo_queue};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
