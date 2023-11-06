// Copyright (C) 2018-2023 Intel Corporation
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
OutputVector translate_partitioned_call_op(const NodeContext& node) {
    default_op_checks(node, 0, {"PartitionedCall", "StatefulPartitionedCall"});
    auto node_name = node.get_name();
    auto translate_session = node.get_translate_session();
    FRONT_END_GENERAL_CHECK(translate_session, "[TensorFlow Frontend] Internal error: Translate session is nullptr.");
    auto operation_type = node.get_attribute<string>("f");

    // prepare a vector of inputs
    OutputVector ov_inputs;
    int input_size = static_cast<int>(node.get_input_size());
    for (int ind = 0; ind < input_size; ++ind) {
        ov_inputs.push_back(node.get_input(ind));
    }

    // try to retrieve ov::Model for body graph
    // Here is a workaround mostly for Saved Model file format. In many tests we use internal name "Identity"
    // to get an output. By default names are cleaned for inputs and outputs because they are available inside
    // of StatefulPartitionedCall. And because otherwise they will cause a duplicates. But we need to keep them
    // for "internal functions of Saved Model", which are named "__inference_signature_wrapper" or
    // "__inference_wrapped_model".
    auto body_model =
        translate_session->get_body_ov_model(operation_type, ov_inputs, operation_type.find("wrappe") == string::npos);
    FRONT_END_OP_CONVERSION_CHECK(
        body_model,
        "[TensorFlow Frontend] Internal error or incorrect input model: body graph is not found for " + operation_type +
            ".");

    // inject the body graph into the parent graph
    OutputVector ov_outputs;
    inject_body_model(body_model, operation_type, ov_inputs, ov_outputs);

    // set output tensor names
    for (size_t idx = 0; idx < ov_outputs.size(); ++idx) {
        set_out_name({node_name + ":" + to_string(idx)}, ov_outputs[idx]);
    }

    return ov_outputs;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
