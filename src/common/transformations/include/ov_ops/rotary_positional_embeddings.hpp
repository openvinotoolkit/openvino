// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

///
/// \brief Rotary Positional Embeddings operation
/// Internal operation which may change in the future
/// \ingroup ov_ops_cpp_api
class TRANSFORMATIONS_API RPE : public ov::op::Op {
public:
    OPENVINO_OP("RPE", "ie_internal_opset", op::Op);

    RPE() = default;
    RPE(const Output<Node>& data, const Output<Node>& sin, const Output<Node>& cos, const int64_t axis);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_axis() const {
        return m_axis;
    };
    void set_axis(const int64_t axis) {
        m_axis = axis;
    };

private:
    int64_t m_axis{};
};

}  // namespace internal
}  // namespace op
}  // namespace ov
