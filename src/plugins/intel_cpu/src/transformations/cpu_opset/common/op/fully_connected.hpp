// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

namespace ov {
namespace intel_cpu {

class FullyConnectedNode : public ngraph::op::Op {
public:
    OPENVINO_OP("FullyConnected", "cpu_plugin_opset");

    FullyConnectedNode() = default;

    FullyConnectedNode(const ov::Output<Node> &A,
                       const ov::Output<Node> &B,
                       const ov::Rank& output_rank,
                       const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ov::Rank get_output_rank() const { return m_output_rank; }
    ov::element::Type get_output_type() const { return m_output_type; }

private:
    ov::Rank m_output_rank;
    ov::element::Type m_output_type;
};

}   // namespace intel_cpu
}   // namespace ov
