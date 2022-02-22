// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class HardSigmoid_IE : public Op {
public:
    OPENVINO_OP("HardSigmoid_IE", "legacy");
    BWDCMP_RTTI_DECLARATION;

    HardSigmoid_IE() = default;

    HardSigmoid_IE(const Output<Node>& arg,
        float alpha,
        float beta);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    float get_alpha() const { return m_alpha; }
    void set_alpha(float alpha) { m_alpha = alpha; }
    float get_beta() const { return m_beta; }
    void set_beta(float beta) { m_beta = beta; }

protected:
    float m_alpha;
    float m_beta;
};

}  // namespace op
}  // namespace ngraph
