// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v16 {
/// \brief PagedAttention operation is used as a placeholder op.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API PagedAttention : public Op {
public:
    OPENVINO_OP("PagedAttention", "opset16");
    PagedAttention() = default;
    /**
     * @brief PagedAttention operation is used as a placeholder. It copies the tensor data to the output.
     */
    PagedAttention(const Output<Node>& data);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void set_out_type(int index, const ov::element::Type& output_type);

protected:
    std::vector<ov::element::Type> m_output_type = {ov::element::undefined, ov::element::undefined};
};
}  // namespace v16
}  // namespace op
}  // namespace ov
