// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/op/util/fused_op.hpp>

namespace MKLDNNPlugin {

class FullyConnectedNode : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"FullyConnected", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }

    FullyConnectedNode() = default;

    FullyConnectedNode(const ngraph::Output<Node> &A,
                       const ngraph::Output<Node> &B,
                       const ngraph::Shape &output_shape,
                       const ngraph::element::Type output_type = ngraph::element::undefined);

    FullyConnectedNode(const ngraph::Output<Node> &A,
                       const ngraph::Output<Node> &B,
                       const ngraph::Output<Node> &C,
                       const ngraph::Shape &output_shape,
                       const ngraph::element::Type output_type = ngraph::element::undefined);

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    size_t get_out_size() const { return m_output_size; }

    ngraph::element::Type get_output_type() const { return m_output_type; }

private:
    size_t m_output_size = 0;
    ngraph::Shape m_output_shape = {};
    ngraph::element::Type m_output_type;
};

}  // namespace MKLDNNPlugin
