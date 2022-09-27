// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v3 {
/// \brief ScatterElementsUpdate operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ScatterElementsUpdate : public Op {
public:
    OPENVINO_OP("ScatterElementsUpdate", "opset3", op::Op, 3);
    BWDCMP_RTTI_DECLARATION;

    ScatterElementsUpdate() = default;
    /// \brief Constructs a ScatterElementsUpdate node

    /// \param data            Input data
    /// \param indices         Data entry index that will be updated
    /// \param updates         Update values
    /// \param axis            Axis to scatter on
    ScatterElementsUpdate(const Output<Node>& data,
                          const Output<Node>& indices,
                          const Output<Node>& updates,
                          const Output<Node>& axis);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;

private:
    bool evaluate_scatter_element_update(const HostTensorVector& outputs, const HostTensorVector& inputs) const;
};
}  // namespace v3
}  // namespace op
}  // namespace ov
