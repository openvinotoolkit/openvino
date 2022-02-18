// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/axis_set.hpp"
#include "openvino/op/util/index_reduction.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Returns embeddings for given indices
class OPENVINO_API EmbeddingBagPackedBase : public Op {
public:
    OPENVINO_OP("EmbeddingBagPackedBase", "util");
    BWDCMP_RTTI_DECLARATION;
    /// \brief Constructs a EmbeddingBagPackedBase operation.
    EmbeddingBagPackedBase() = default;
    /// \brief Constructs a EmbeddingBagPackedBase operation.
    ///
    /// EmbeddingBagPackedBase constructs an output tensor by replacing every index in a
    /// given
    /// input tensor with a row (from the weights matrix) at that index
    ///
    /// \param emb_table Tensor containing the embedding lookup table of the module of
    /// shape [num_emb, emb_dim1, emb_dim2, ...] and  of type T
    /// \param  indices Tensor of shape `[batch, indices_per_bag]` and of type *T_IND*.
    /// Required.
    /// \param per_sample_weigths tensor of the same shape as indices and of type T.
    /// Each value in this tensor are multiplied with each
    /// value pooled from embedding table for each index. Optional.

    EmbeddingBagPackedBase(const Output<Node>& emb_table,
                           const Output<Node>& indices,
                           const Output<Node>& per_sample_weights);

    EmbeddingBagPackedBase(const Output<Node>& emb_table, const Output<Node>& indices);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    static constexpr int EMB_TABLE = 0;
    static constexpr int INDICES = 1;
    static constexpr int PER_SAMPLE_WEIGHTS = 2;
};
}  // namespace util
}  // namespace op
}  // namespace ov
