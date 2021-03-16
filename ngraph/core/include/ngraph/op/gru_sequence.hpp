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

#include <memory>
#include <string>
#include <vector>

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/rnn_cell_base.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v5
        {
            class NGRAPH_API GRUSequence : public util::RNNCellBase
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                GRUSequence();

                GRUSequence(const Output<Node>& X,
                            const Output<Node>& H_t,
                            const Output<Node>& sequence_lengths,
                            const Output<Node>& W,
                            const Output<Node>& R,
                            const Output<Node>& B,
                            size_t hidden_size,
                            op::RecurrentSequenceDirection direction,
                            const std::vector<std::string>& activations =
                                std::vector<std::string>{"sigmoid", "tanh"},
                            const std::vector<float>& activations_alpha = {},
                            const std::vector<float>& activations_beta = {},
                            float clip = 0.f,
                            bool linear_before_reset = false);

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                void validate_and_infer_types() override;

                bool visit_attributes(AttributeVisitor& visitor) override;
                bool get_linear_before_reset() const { return m_linear_before_reset; }
                op::RecurrentSequenceDirection get_direction() const { return m_direction; }

            protected:
                op::RecurrentSequenceDirection m_direction;
                bool m_linear_before_reset;
            };
        }
    } // namespace op
} // namespace ngraph
