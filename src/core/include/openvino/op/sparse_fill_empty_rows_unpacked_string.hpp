// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::v16 {
/// \brief An operation which fills empty rows of a sparse string tensor with a default string value.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SparseFillEmptyRowsUnpackedString : public ov::op::Op {
public:
    OPENVINO_OP("SparseFillEmptyRowsUnpackedString", "opset16");

    SparseFillEmptyRowsUnpackedString() = default;

    /// \brief Constructs a SparseFillEmptyRowsUnpackedString operation.
    ///
    /// \param begins 1D tensor containing the beginning indices of strings in the symbols array.
    /// \param ends 1D tensor containing the ending indices of strings in the symbols array.
    /// \param symbols 1D tensor containing the concatenated string data encoded in utf-8 bytes.
    /// \param indices 2D tensor indicating the positions at which values are placed in the sparse tensor.
    /// \param dense_shape 1D tensor indicating the shape of the dense tensor.
    /// \param default_value A string scalar to be inserted into the empty rows.
    SparseFillEmptyRowsUnpackedString(const Output<Node>& begins,
                                      const Output<Node>& ends,
                                      const Output<Node>& symbols,
                                      const Output<Node>& indices,
                                      const Output<Node>& dense_shape,
                                      const Output<Node>& default_value);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace ov::op::v16
