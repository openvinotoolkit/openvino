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

    FullyConnectedNode(const ngraph::Output<Node> &A,
                       const ngraph::Output<Node> &B,
                       const ngraph::Rank& output_rank,
                       const ngraph::element::Type output_type = ngraph::element::undefined);

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    ngraph::Rank get_output_rank() const { return m_output_rank; }
    ngraph::element::Type get_output_type() const { return m_output_type; }

private:
    ngraph::Rank m_output_rank;
    ngraph::element::Type m_output_type;
};

}   // namespace intel_cpu
}   // namespace ov
