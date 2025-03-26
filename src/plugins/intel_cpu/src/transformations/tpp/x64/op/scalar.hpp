// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "eltwise.hpp"
#include "snippets/op/reduce.hpp"
#include "transformations/tpp/common/op/modifiers.hpp"

namespace ov::intel_cpu::tpp::op {
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

}  // namespace ov::intel_cpu::tpp::op
