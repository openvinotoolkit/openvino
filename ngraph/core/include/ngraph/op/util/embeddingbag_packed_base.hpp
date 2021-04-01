//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/op/util/index_reduction.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            /// \brief Returns embeddings for given indices
            class NGRAPH_API EmbeddingBagPackedBase : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"EmbeddingBagPackedBase", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
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
        }
        using util::EmbeddingBagPackedBase;
    }
}
