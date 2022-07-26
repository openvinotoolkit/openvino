// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/scatter_nd_base.hpp"

namespace ov {
namespace op {
namespace v3 {
/// \brief Add updates to slices from inputs addressed by indices
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ScatterNDUpdate : public util::ScatterNDBase {
public:
    OPENVINO_OP("ScatterNDUpdate", "opset4", util::ScatterNDBase, 3);
    BWDCMP_RTTI_DECLARATION;
    ScatterNDUpdate() = default;
    /// \param inputs Tensor
    /// \param indices Index tensor: Data type must be `element::i32` or `element::i64`
    /// \param updates Tensor: Must have same type as inputs
    ScatterNDUpdate(const Output<Node>& inputs, const Output<Node>& indices, const Output<Node>& updates)
        : util::ScatterNDBase(inputs, indices, updates) {}

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;
};
}  // namespace v3
}  // namespace op
}  // namespace ov
