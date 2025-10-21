// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "openvino/core/model_util.hpp"
#include "openvino/core/graph_util.hpp"

namespace ov::util {

class PostponedConstant : public ov::op::Op {
public:
    OPENVINO_OP("Constant", "opset1");

    PostponedConstant(std::shared_ptr<ov::Node> node) : m_node(std::move(node)) {
        OPENVINO_ASSERT(m_node);
        OPENVINO_ASSERT(m_node->get_output_size() == 1);
        constructor_validate_and_infer_types();
    };

    void validate_and_infer_types() {
        set_output_size(1);
        auto&& output = m_node->output(0);
        set_output_type(0, output.get_element_type(), output.get_partial_shape());
    }

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& inputs) const {
        OPENVINO_THROW("PostponedConstant cannot be copied");
    }

    bool visit_attributes(AttributeVisitor& visitor) {
        ov::OutputVector outputs(1);
        OPENVINO_ASSERT(
            m_node->constant_fold(outputs, m_node->input_values()),
            "Node with set `postponed_constant` attribute cannot be fold to constant when saving model to IR file");
        return outputs[0].get_node_shared_ptr()->visit_attributes(visitor);
    }
private:
    std::shared_ptr<ov::Node> m_node;
};


std::shared_ptr<ov::Node> make_postponed_constant_from_node(std::shared_ptr<ov::Node> node) {
    auto postponed_constant = std::make_shared<PostponedConstant>(node);
    ov::replace_node(node, postponed_constant);
    return postponed_constant;
}

}  // namespace ov::util
