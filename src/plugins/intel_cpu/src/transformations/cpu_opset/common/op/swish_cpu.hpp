// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace intel_cpu {

class SwishNode : public ov::op::Op {
public:
    OPENVINO_OP("SwishCPU", "cpu_plugin_opset");

    SwishNode() = default;

    explicit SwishNode(const ov::Output<Node> &input, float alpha = 1.0);

    void validate_and_infer_types() override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector &new_args) const override;

    float get_alpha() const;

protected:
    float m_alpha;
};

}   // namespace intel_cpu
}   // namespace ov
