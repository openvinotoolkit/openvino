// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v15 {
/// \brief Identity operation is used as a placeholder op.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Identity : public Op {
public:
    OPENVINO_OP("Identity", "opset15");
    Identity() = default;
    /**
     * @brief Identity operation is used as a placeholder. It either passes the tensor down to the next layer,
     * or copies the tensor to the output.
     *
     * @param copy Boolean that determines whether to copy the input to the output, or just return the output.
     */
    Identity(const Output<Node>& data, const bool copy = false);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool get_copy() const;
    void set_copy(const bool copy);

private:
    bool m_copy;
};
}  // namespace v15
}  // namespace op
}  // namespace ov