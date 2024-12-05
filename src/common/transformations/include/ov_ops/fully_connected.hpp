// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

class TRANSFORMATIONS_API FullyConnected : public ov::op::Op {
public:
    OPENVINO_OP("FullyConnected", "ie_internal_opset");

    FullyConnected() = default;

    FullyConnected(const OutputVector& arguments, const ov::element::Type output_type = ov::element::undefined);

    FullyConnected(const ov::Output<Node>& A,
                   const ov::Output<Node>& B,
                   const ov::Output<Node>& bias,
                   const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    virtual std::shared_ptr<Node> fuse_bias(const ov::Output<Node>& bias) const;

    ov::element::Type get_output_type() const {
        return m_output_type;
    }

protected:
    ov::element::Type m_output_type;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
