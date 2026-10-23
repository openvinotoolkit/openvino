// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

namespace swoosh_constants {
inline constexpr float slope = 0.08f;
inline constexpr float swoosh_l_offset = 4.0f;
inline constexpr float swoosh_l_bias = 0.035f;
inline constexpr float swoosh_r_offset = 1.0f;
inline constexpr float swoosh_r_bias = 0.3132616877555847f;
}  // namespace swoosh_constants

class Swoosh : public ov::op::Op {
public:
    OPENVINO_OP("Swoosh", "ie_internal_opset");

    Swoosh() = default;
    Swoosh(const ov::Output<Node>& data, float offset, float bias);

    void validate_and_infer_types() override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    float get_offset() const {
        return m_offset;
    }

    float get_bias() const {
        return m_bias;
    }

private:
    float m_offset = 0.0f;
    float m_bias = 0.0f;
};

}  // namespace ov::intel_gpu::op
