// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "input_model.hpp"
#include "tf_utils.hpp"

using namespace std;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_while_op(const NodeContext& node) {
    default_op_checks(node, 1, {"While", "StatelessWhile"});
    auto node_name = node.get_name();
    auto translate_session = node.get_translate_session();
    auto input_size_t = node.get_input_size();
    auto input_size = static_cast<int>(input_size_t);

    OutputVector ov_inputs;
    for (int input_ind = 0; input_ind < input_size; ++input_ind) {
        ov_inputs.push_back(node.get_input(input_ind));
    }

    TENSORFLOW_OP_VALIDATION(node,
                             translate_session,
                             "[TensorFlow Frontend] Internal error: Translate session is nullptr.");
    // retrieve condition and body graphs
    auto cond_type = node.get_attribute<string>("cond");
    auto body_type = node.get_attribute<string>("body");
    auto cond_model = translate_session->get_body_ov_model(cond_type, ov_inputs);
    TENSORFLOW_OP_VALIDATION(
        node,
        cond_model,
        "[TensorFlow Frontend] Internal error or incorrect input model. Cannot find body graph with name " + cond_type);
    auto body_model = translate_session->get_body_ov_model(body_type, ov_inputs);
    TENSORFLOW_OP_VALIDATION(
        node,
        body_model,
        "[TensorFlow Frontend] Internal error or incorrect input model. Cannot find body graph with name " + body_type);

    auto loop = create_loop_for_tf_while(node.get_name(), body_model, cond_model, ov_inputs);
    set_node_name(node.get_name(), loop);
    return loop->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
