// Copyright (C) 2018-2025 Intel Corporation
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
    auto data_producer = data.get_node_shared_ptr();

    // check if the condition goes to another Switch node
    // and re-use its conditional flow marker for further grouping multiple Switch and Merge nodes
    // walk through all consumers for pred and find Switch nodes
    bool generate_new_marker = true;
    int32_t switch_marker = 0;
    auto pred_node = pred.get_node_shared_ptr();
    auto pred_index = pred.get_index();
    bool to_skip_switch = false;
    for (const auto& target_input : pred_node->get_output_target_inputs(pred_index)) {
        auto another_producer = target_input.get_node()->shared_from_this();
        if (const auto& another_switch = as_type_ptr<Switch>(another_producer)) {
            if (another_switch->is_cond_flow_marker_set() && cf_marker_exists(another_switch)) {
                switch_marker = another_switch->get_cond_flow_marker();
                generate_new_marker = false;
            }

            // in case two consecutive Switch nodes with the common predicate
            // the successor can be skipped
            if (another_switch == data_producer) {
                to_skip_switch = true;
                generate_new_marker = false;
                break;
            }
        }
    }

    if (generate_new_marker) {
        // generate marker for new conditioning flow
        switch_marker = generate_cf_marker();
    }
    auto switch_node = make_shared<Switch>(data, pred, switch_marker, node.get_decoder());

    // in case two consecutive Switch nodes with the common predicate
    // it skips successive Switch node
    if (to_skip_switch) {
        auto data_output_index = data.get_index();
        FRONT_END_GENERAL_CHECK(data_output_index == 0 || data_output_index == 1,
                                "[TensorFlow Frontend] internal error: Switch node must have two outputs");
        size_t other_output_index = 1 - data_output_index;
        data.add_names({switch_name + ":" + to_string(data_output_index)});
        auto other_switch_output = switch_node->output(other_output_index);
        other_switch_output.add_names({switch_name + ":" + to_string(other_output_index)});
        if (data_output_index == 0) {
            return OutputVector{data, other_switch_output};
        } else {
            return OutputVector{other_switch_output, data};
        }
    }

    // set output tensor names and move the node name
    set_node_name(switch_name, switch_node);
    return switch_node->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
