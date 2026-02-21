// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {

/// \brief SparseTypeMark serves to mark sparse tensor representations.
///
/// This marker defers the conversion from sparse to dense, allowing
/// operations to work directly with sparse data (indices, values).
///
/// Represents:
/// - indices: [ndim, nnz] or [nnz, ndim]
/// - values: [nnz]
/// - shape: [ndim]
///
/// Usage:
///   auto sparse_mark = std::make_shared<SparseTypeMark>(indices, values, shape);
///   auto result = SparseTypeMark::sparse_mm(context, sparse_mark, dense);
///
class FRONTEND_API SparseTypeMark : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("SparseTypeMark", "util", ov::op::util::FrameworkNode);

    /// \brief Construct a SparseTypeMark from sparse COO representation
    /// \param indices Tensor of indices, shape [ndim, nnz] or [nnz, ndim]
    /// \param values Tensor of values at those indices, shape [nnz]
    /// \param shape Tensor describing the dense shape, shape [ndim]
    /// \param value_type Element type of the values
    SparseTypeMark(const ov::Output<ov::Node>& indices,
                   const ov::Output<ov::Node>& values,
                   const ov::Output<ov::Node>& shape,
                   const ov::element::Type& value_type = ov::element::dynamic);

    ~SparseTypeMark() override;

    void validate_and_infer_types() override {
        set_output_type(0, ov::element::dynamic, PartialShape::dynamic());
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        OPENVINO_ASSERT(inputs.size() == 3, "SparseTypeMark expects 3 inputs");
        auto sparse_mark = std::make_shared<SparseTypeMark>(inputs[0], inputs[1], inputs[2], m_value_type);
        sparse_mark->set_attrs(get_attrs());
        return sparse_mark;
    }

    /// \brief Get the element type of the sparse tensor values
    ov::element::Type get_value_type() const {
        return m_value_type;
    }

    /// \brief Get the indices tensor (lazy accessor)
    /// \return Indices tensor, shape [ndim, nnz] or [nnz, ndim]
    ov::Output<ov::Node> get_indices();

    /// \brief Get the values tensor (lazy accessor)
    /// \return Values tensor, shape [nnz]
    ov::Output<ov::Node> get_values();

    /// \brief Get the shape tensor (lazy accessor)
    /// \return Shape tensor, shape [ndim]
    ov::Output<ov::Node> get_shape();

    /// \brief Convert sparse representation to dense tensor
    /// This is the fallback for operations that don't have sparse-aware implementations
    /// \param context Node context for creating operations
    /// \return Dense tensor with the same values as the sparse representation
    ov::Output<ov::Node> to_dense(const NodeContext& context);

    /// \brief Sparse-aware matrix multiplication: sparse @ dense -> dense
    /// \param context Node context for creating operations
    /// \param sparse Sparse matrix (SparseTypeMark or dense)
    /// \param dense Dense matrix
    /// \return Result of matrix multiplication
    static ov::Output<ov::Node> sparse_mm(const NodeContext& context,
                                          const ov::Output<ov::Node>& sparse,
                                          const ov::Output<ov::Node>& dense);

    /// \brief Sparse-aware addition: sparse + sparse/dense -> sparse/dense
    /// \param context Node context for creating operations
    /// \param lhs Left operand (SparseTypeMark or dense)
    /// \param rhs Right operand (SparseTypeMark or dense)
    /// \return Result of addition
    static ov::Output<ov::Node> sparse_add(const NodeContext& context,
                                           const ov::Output<ov::Node>& lhs,
                                           const ov::Output<ov::Node>& rhs);

private:
    ov::Output<ov::Node> m_indices;
    ov::Output<ov::Node> m_values;
    ov::Output<ov::Node> m_shape;
    ov::element::Type m_value_type;

    // Cached dense representation (lazy, only created when needed)
    mutable ov::Output<ov::Node> m_dense;
};

}  // namespace frontend
}  // namespace ov
