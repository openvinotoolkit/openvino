// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Dynamic Tiling operation which repeats a tensor multiple times
///        along each dimension
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Tile : public Op {
public:
    OPENVINO_OP("Tile", "opset1");
    BWDCMP_RTTI_DECLARATION;

    Tile() = default;
    /// \brief Perform dynamic padding of a tensor
    ///
    /// \param data The node producing input tensor to be padded.
    /// \param repeats The node producing the per-dimension replication factor
    Tile(const Output<Node>& data, const Output<Node>& repeats);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;

private:
    bool evaluate_tile(const HostTensorVector& outputs, const HostTensorVector& inputs) const;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
