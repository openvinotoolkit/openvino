// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace MKLDNNPlugin {

class SwishNode : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"SwishCPU", 0};
    static constexpr const ::ngraph::Node::type_info_t& get_type_info_static() { return type_info; }
    const ngraph::NodeTypeInfo &get_type_info() const override { return type_info; }

    SwishNode() = default;

    explicit SwishNode(const ngraph::Output<Node> &input, float alpha = 1.0);

    void validate_and_infer_types() override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector &new_args) const override;

    float get_alpha() const;

protected:
    float m_alpha;
};

}  // namespace MKLDNNPlugin
