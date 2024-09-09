// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/layout.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Decode JPEG.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API DecodeImg : public Op {
public:
    OPENVINO_OP("DecodeImg", "opset1");

    /// \brief Constructs a cosine operation.
    DecodeImg() = default;
    /// \brief Constructs a cosine operation.
    ///
    /// \param arg Node that produces the input tensor.
    DecodeImg(const Output<Node>& arg);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

protected:
    int m_width;
    int m_height;
    int m_channel;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
