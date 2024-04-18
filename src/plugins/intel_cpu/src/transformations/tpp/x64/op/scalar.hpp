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
// Note that the tpp::op::Scalar is not derived from the TensorProcessingPrimitive modifier. We don't need it because
// the Scalar is not a MemoryAccess operation, since it doesn't need to read from the external
// memory, and hence it is not really a TPP.
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
