// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <transformations_visibility.hpp>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
namespace internal {
class FakeDequantInternal : public Op {
public:
    OPENVINO_OP("FakeDequantInternal", "util");
    BWDCMP_RTTI_DECLARATION;

    FakeDequantInternal() = default;

    FakeDequantInternal(const Output<Node>& x,
                      const Output<Node>& scale,
                      const int bit_length,
                      const float max_range);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & new_args) const override;

    int m_bit_length = 0;
    float m_max_range = 0.0f;

private:
};

}  // namespace internal
}  // namespace op
}  // namespace ngraph
