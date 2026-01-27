// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sparse_type_mark.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/transpose.hpp"

using namespace ov::frontend;
using namespace ov::op;
using namespace std;

SparseTypeMark::SparseTypeMark(const ov::Output<ov::Node>& indices,
                               const ov::Output<ov::Node>& values,
                               const ov::Output<ov::Node>& shape,
                               const ov::element::Type& value_type)
    : ov::op::util::FrameworkNode(ov::OutputVector{indices, values, shape}, 1),
      m_indices(indices),
      m_values(values),
      m_shape(shape),
      m_value_type(value_type),
      m_dense{} {
    validate_and_infer_types();

    if (m_value_type.is_dynamic()) {
        m_value_type = m_values.get_element_type();
    }
}

SparseTypeMark::~SparseTypeMark() = default;

ov::Output<ov::Node> SparseTypeMark::get_indices() {
    return m_indices;
}

ov::Output<ov::Node> SparseTypeMark::get_values() {
    return m_values;
}

ov::Output<ov::Node> SparseTypeMark::get_shape() {
    return m_shape;
}

ov::Output<ov::Node> SparseTypeMark::to_dense(const NodeContext& context) {
    // Return cached dense representation
    if (m_dense.get_node_shared_ptr()) {
        return m_dense;
    }

    // Create dense tensor
    auto zero_const = context.mark_node(v0::Constant::create(m_value_type, Shape{}, {0}));
    auto dense_zero = context.mark_node(make_shared<v3::Broadcast>(zero_const, m_shape));

    // Indices are [ndim, nnz], transpose to [nnz, ndim] for ScatterNDUpdate
    auto perm_order = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, 0}));
    auto permuted_indices = context.mark_node(make_shared<v1::Transpose>(m_indices, perm_order));

    // Convert to dense
    m_dense = context.mark_node(make_shared<v3::ScatterNDUpdate>(dense_zero, permuted_indices, m_values));

    return m_dense;
}

ov::Output<ov::Node> SparseTypeMark::sparse_mm(const NodeContext& context,
                                               const ov::Output<ov::Node>& sparse,
                                               const ov::Output<ov::Node>& matrix) {
    auto sparse_mark = as_type_ptr<SparseTypeMark>(sparse.get_node_shared_ptr());
    auto matrix_mark = as_type_ptr<SparseTypeMark>(matrix.get_node_shared_ptr());

    // Both inputs are sparse (S x S -> S in PyTorch, but we emulate with dense for now)
    if (sparse_mark && matrix_mark) {
        auto lhs_dense = sparse_mark->to_dense(context);
        auto rhs_dense = matrix_mark->to_dense(context);
        return context.mark_node(make_shared<v0::MatMul>(lhs_dense, rhs_dense, false, false));
    }

    // Only first input is sparse (S x D -> D)
    if (sparse_mark) {
        auto dense_sparse = sparse_mark->to_dense(context);
        return context.mark_node(make_shared<v0::MatMul>(dense_sparse, matrix, false, false));
    }

    // Note: D x S case is not explicitly handled because PyTorch's torch.sparse.mm
    // strictly supports (Sparse x Sparse -> Sparse) and (Sparse x Dense -> Dense).
    // Dense x Sparse is not part of the standard sparse contract for this operator.

    // Neither is sparse-marked
    return context.mark_node(make_shared<v0::MatMul>(sparse, matrix, false, false));
}

ov::Output<ov::Node> SparseTypeMark::sparse_add(const NodeContext& context,
                                                const ov::Output<ov::Node>& lhs,
                                                const ov::Output<ov::Node>& rhs) {
    auto lhs_sparse = as_type_ptr<SparseTypeMark>(lhs.get_node_shared_ptr());
    auto rhs_sparse = as_type_ptr<SparseTypeMark>(rhs.get_node_shared_ptr());

    if (lhs_sparse && rhs_sparse) {
        // Both sparse: merge indices/values (future optimization)
        // For now, convert both to dense
        auto lhs_dense = lhs_sparse->to_dense(context);
        auto rhs_dense = rhs_sparse->to_dense(context);
        return context.mark_node(make_shared<v1::Add>(lhs_dense, rhs_dense));
    } else if (lhs_sparse) {
        // Only lhs is sparse
        auto lhs_dense = lhs_sparse->to_dense(context);
        return context.mark_node(make_shared<v1::Add>(lhs_dense, rhs));
    } else if (rhs_sparse) {
        // Only rhs is sparse
        auto rhs_dense = rhs_sparse->to_dense(context);
        return context.mark_node(make_shared<v1::Add>(lhs, rhs_dense));
    }

    // Both dense
    return context.mark_node(make_shared<v1::Add>(lhs, rhs));
}
