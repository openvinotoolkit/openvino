// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
namespace internal {
class FakeQuantDequantInternal : public Op {
public:
    OPENVINO_OP("FakeQuantDequantInternal", "util");
    BWDCMP_RTTI_DECLARATION;

    FakeQuantDequantInternal() = default;

    FakeQuantDequantInternal(const Output<Node>& x,
                             const Output<Node>& scale,
                             const std::string& op_type,
                             const int quant_axis,
                             const int bit_length = 8);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    std::string m_op_type;
    int m_quant_axis = 0;
    int m_bit_length = 8;

private:
};

}  // namespace internal
}  // namespace op
}  // namespace ngraph
