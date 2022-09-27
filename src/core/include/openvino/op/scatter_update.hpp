// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/scatter_base.hpp"

namespace ov {
namespace op {
namespace v3 {
///
/// \brief      Set new values to slices from data addressed by indices
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API ScatterUpdate : public util::ScatterBase {
public:
    OPENVINO_OP("ScatterUpdate", "opset3", util::ScatterBase, 3);
    BWDCMP_RTTI_DECLARATION;
    ScatterUpdate() = default;
    ///
    /// \brief      Constructs ScatterUpdate operator object.
    ///
    /// \param      data     The input tensor to be updated.
    /// \param      indices  The tensor with indexes which will be updated.
    /// \param      updates  The tensor with update values.
    /// \param[in]  axis     The axis at which elements will be updated.
    ///
    ScatterUpdate(const Output<Node>& data,
                  const Output<Node>& indices,
                  const Output<Node>& updates,
                  const Output<Node>& axis);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
    bool has_evaluate() const override;

private:
    bool evaluate_scatter_update(const HostTensorVector& outputs, const HostTensorVector& inputs) const;
};
}  // namespace v3
}  // namespace op
}  // namespace ov
