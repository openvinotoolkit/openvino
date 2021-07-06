// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/op/util/embeddingbag_packed_base.hpp"
#include "ngraph/op/util/index_reduction.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v3
        {
            /// \brief Returns embeddings for given indices
            class NGRAPH_API EmbeddingBagPackedSum : public util::EmbeddingBagPackedBase
            {
            public:
                static constexpr NodeTypeInfo type_info{"EmbeddingBagPackedSum", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a EmbeddingBagPackedSum operation.
                EmbeddingBagPackedSum() = default;
                /// \brief Constructs a EmbeddingBagPackedSum operation.
                ///
                /// EmbeddingBagPackedSum constructs an output tensor by replacing every index in a
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

                EmbeddingBagPackedSum(const Output<Node>& emb_table,
                                      const Output<Node>& indices,
                                      const Output<Node>& per_sample_weights);

                EmbeddingBagPackedSum(const Output<Node>& emb_table, const Output<Node>& indices);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v3
        using v3::EmbeddingBagPackedSum;
    } // namespace op
} // namespace ngraph
