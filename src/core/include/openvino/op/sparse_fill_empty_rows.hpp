// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::v16 {
/// \brief An operation which fills empty rows of a sparse tensor with a default value.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SparseFillEmptyRows : public ov::op::Op {
public:
    OPENVINO_OP("SparseFillEmptyRows", "opset16");

    SparseFillEmptyRows() = default;

    /// \brief Constructs a SparseFillEmptyRows operation.
    ///
    /// \param indices 2D tensor indicating the positions of values in the sparse tensor.
    /// \param values 1D tensor containing the values to be inserted at the specified indices.
    /// \param dense_shape 1D tensor indicating the shape of the 2D dense tensor.
    /// \param default_value Scalar value to be inserted into the empty rows.
    SparseFillEmptyRows(const Output<Node>& indices,
                        const Output<Node>& values,
                        const Output<Node>& dense_shape,
                        const Output<Node>& default_value);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace ov::op::v16
