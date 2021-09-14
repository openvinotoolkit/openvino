// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace MKLDNNPlugin {

class LeakyReluNode : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"LeakyRelu", 0};
    static constexpr const ::ngraph::Node::type_info_t& get_type_info_static() { return type_info; }
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }

    LeakyReluNode() = default;

    LeakyReluNode(const ngraph::Output<ngraph::Node> &data, const float &negative_slope, const ngraph::element::Type output_type);

    void validate_and_infer_types() override;

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector &new_args) const override;

    float get_slope() { return m_negative_slope; }

    ngraph::element::Type get_output_type() const { return m_output_type; }

private:
    float m_negative_slope;
    ngraph::element::Type m_output_type;
};

}  // namespace MKLDNNPlugin
