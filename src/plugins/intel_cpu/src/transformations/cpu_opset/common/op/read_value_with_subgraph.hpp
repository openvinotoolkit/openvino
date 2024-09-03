// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

namespace ov {
namespace intel_cpu {

class ReadValueWithSubgraphNode : public op::util::MultiSubGraphOp {
public:
    OPENVINO_OP("ReadValueWithSubgraph", "cpu_plugin_opset");

    ReadValueWithSubgraphNode();
    ReadValueWithSubgraphNode(const std::shared_ptr<ov::op::util::Variable>& variable);

    void set_body(const std::shared_ptr<Model>& body) {
        m_bodies[0] = body;
    }
    const std::shared_ptr<Model>& get_body() const {
        return m_bodies[0];
    }

    void set_input(const Output<Node>& value, const std::shared_ptr<op::v0::Parameter>& body_parameter);

    Output<Node> set_output(const std::shared_ptr<op::v0::Result>& body_result);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
private:
    std::shared_ptr<Model> m_subgraph;
    std::shared_ptr<op::util::Variable> m_variable;
};

}   // namespace intel_cpu
}   // namespace ov
