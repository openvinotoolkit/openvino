// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "modifiers.hpp"
#include "eltwise.hpp"
#include "snippets/op/reduce.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

class Scalar : public ov::snippets::op::Scalar {
public:
    OPENVINO_OP("Scalar", "TppOpset", snippets::op::Scalar);

    Scalar() = default;
    explicit Scalar(const snippets::op::Scalar& other);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
