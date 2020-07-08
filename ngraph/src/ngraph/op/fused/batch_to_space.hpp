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

#include "ngraph/node.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief BatchToSpace permutes data from the batch dimension of the data tensor into
            ///        spatial dimensions.
            ///
            /// \note  Values from the batch dimension are moved in spatial blocks dimensions.
            ///
            ///        Output node produces a tensor with shape:
            ///        `[batch / (block_shape[0] * block_shape[1] * ... * block_shape[N - 1]),
            ///         D_1 * block_shape[1] - crops_begin[1] - crops_end[1],
            ///         D_2 * block_shape[2] - crops_begin[2] - crops_end[2], ...,
            ///         D_{N - 1} * block_shape[N - 1] - crops_begin[N - 1] - crops_end[N - 1]`
            ///         of the same type as `data` input.
            class NGRAPH_API BatchToSpace : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"BatchToSpace", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                BatchToSpace() = default;
                /// \brief Constructs a BatchToSpace operation.
                ///
                /// \param data Node producing the data tensor
                /// \param block_shape The sizes of the block of values to be moved
                /// \param crops_begin Specifies the amount to crop from the beginning along each
                /// axis of `data` input
                /// \param crops_end Specifies the amount to crop from the ending along each axis of
                /// `data` input.
                BatchToSpace(const Output<Node>& data,
                             const Output<Node>& block_shape,
                             const Output<Node>& crops_begin,
                             const Output<Node>& crops_end);

                OutputVector decompose_op() const override;
                void pre_validate_and_infer_types() override;
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool visit_attributes(AttributeVisitor& visitor) override;
            };
        }
    }
}
