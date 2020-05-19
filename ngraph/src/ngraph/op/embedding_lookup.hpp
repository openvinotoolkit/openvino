//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
        namespace v0
        {
            /// \brief Returns embeddings for given indices
            class NGRAPH_API EmbeddingLookup : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"EmbeddingLookup", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a EmbeddingLookup operation.
                EmbeddingLookup() = default;
                /// \brief Constructs a EmbeddingLookup operation.
                ///
                /// EmbeddingLookup constructs an output tensor by replacing every index in a given
                /// input tensor with a row (from the weights matrix) at that index
                ///
                /// \param data The input indices for tokens to be translated into embeddings
                /// \param weights is a dense matrix [N,M] where each row 0..N
                /// corresponds to an embedding (i.e. typically, a vector of real numbers) of length
                /// M
                EmbeddingLookup(const Output<Node>& data, const Output<Node>& weights)
                    : Op({data, weights})
                {
                    constructor_validate_and_infer_types();
                }

                void validate_and_infer_types() override;

                void generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                       const OutputVector& /* deltas */) override
                {
                    throw ngraph_error("Not yet implemented");
                }
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }
        using v0::EmbeddingLookup;
    }
}
