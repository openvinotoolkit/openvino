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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Generates the complete beams from the ids per each step and the parent beam
            /// ids.
            class NGRAPH_API GatherTree : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"GatherTree", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                GatherTree() = default;
                /// \param step_ids     Tensor of shape [MAX_TIME, BATCH_SIZE, BEAM_WIDTH] with
                ///                     indices from per each step
                /// \param parent_idx   Tensor of shape [MAX_TIME, BATCH_SIZE, BEAM_WIDTH] with
                ///                     parent beam indices
                /// \param max_seq_len  Tensor of shape [BATCH_SIZE] with maximum lengths for each
                ///                     sequence in the batch
                /// \param end_token    Tensor of shape [MAX_TIME, BATCH_SIZE, BEAM_WIDTH]
                GatherTree(const Output<Node>& step_ids,
                           const Output<Node>& parent_idx,
                           const Output<Node>& max_seq_len,
                           const Output<Node>& end_token);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }
    }
}
