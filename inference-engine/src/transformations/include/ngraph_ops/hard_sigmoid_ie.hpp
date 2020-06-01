// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class TRANSFORMATIONS_API HardSigmoid_IE : public Op {
public:
    static constexpr NodeTypeInfo type_info{"HardSigmoid_IE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    HardSigmoid_IE() = default;

    HardSigmoid_IE(const Output<Node>& arg,
        float alpha,
        float beta);

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;
    void validate_and_infer_types() override;

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
