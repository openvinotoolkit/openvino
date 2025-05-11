// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/unsupported_constant.hpp"
#include "input_model.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "translate_session.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_variable_op(const NodeContext& node) {
    default_op_checks(node, 0, {"Variable"});
    auto variable_name = node.get_name();

    auto translate_session = node.get_translate_session();
    TENSORFLOW_OP_VALIDATION(node,
                             translate_session,
                             "[TensorFlow Frontend] Internal error: Translate session is nullptr.");
    auto model = dynamic_pointer_cast<ov::frontend::tensorflow::InputModel>(translate_session->get_input_model());
    TENSORFLOW_OP_VALIDATION(
        node,
        model,
        "[TensorFlow Frontend] Internal error: input model is unable to cast to TensorFlow Frontend InputModel.");
    auto checkpoint_v1_reader = model->get_checkpoint_v1_reader();
    TENSORFLOW_OP_VALIDATION(node,
                             checkpoint_v1_reader,
                             "[TensorFlow Frontend] incorrect input model: checkpoint to restore variable " +
                                 variable_name + " is not provided.");

    ov::Any variable_data;
    checkpoint_v1_reader->read_variable(variable_name, variable_data);

    shared_ptr<Node> const_node = nullptr;
    if (variable_data.is<ov::Tensor>()) {
        auto ov_tensor = variable_data.as<ov::Tensor>();
        const_node = make_shared<v0::Constant>(ov_tensor);
    } else {
        // data of unknown type
        auto const_node = make_shared<UnsupportedConstant>("Variable of unsupported type", node.get_decoder());
    }

    set_node_name(variable_name, const_node);
    return {const_node};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
