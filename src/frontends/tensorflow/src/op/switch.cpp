// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/switch.hpp"

#include "common_op_table.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_switch_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Switch"});
    auto switch_name = node.get_name();
    auto data = node.get_input(0);
    auto pred = node.get_input(1);

    // check if the condition goes to another Switch node
    // and re-use its conditional flow marker for further grouping multiple Switch and Merge nodes
    // walk through all consumers for pred and find Switch nodes
    bool another_switch_found = false;
    int32_t switch_marker = 0;
    auto pred_node = pred.get_node_shared_ptr();
    auto pred_index = pred.get_index();
    pred_node->get_output_target_inputs(pred_index);
    for (const auto& target_input : pred_node->get_output_target_inputs(pred_index)) {
        auto another_producer = target_input.get_node()->shared_from_this();
        if (auto another_switch = ov::as_type_ptr<Switch>(another_producer)) {
            if (another_switch->is_cond_flow_marker_set()) {
                switch_marker = another_switch->get_cond_flow_marker();
                another_switch_found = true;
                break;
            }
        }
    }

    if (!another_switch_found) {
        // generate marker for new conditioning flow
        switch_marker = generate_cf_marker();
    }

    auto switch_node = make_shared<Switch>(data, pred, switch_marker, node.get_decoder());

    // set the same marker as for other Switch nodes with the same predicate
    set_node_name(switch_name, switch_node);

    return switch_node->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
