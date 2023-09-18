// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class RMS : public ov::op::Op {
public:
    OPENVINO_OP("RMS", "gpu_opset");

    RMS() = default;

    RMS(const Output<Node>& data,
        const Output<Node>& gamma,
        double epsilson);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    double get_epsilon() const { return m_epsilon; }

    void set_epsilon(double epsilon) { m_epsilon = epsilon; }

private:
    double m_epsilon{0};
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
