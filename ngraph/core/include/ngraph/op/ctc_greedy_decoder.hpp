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
        namespace v0
        {
            class NGRAPH_API CTCGreedyDecoder : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"CTCGreedyDecoder", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                CTCGreedyDecoder() = default;
                /// \brief Constructs a CTCGreedyDecoder operation
                ///
                /// \param input              Logits on which greedy decoding is performed
                /// \param seq_len            Sequence lengths
                /// \param ctc_merge_repeated Whether to merge repeated labels
                CTCGreedyDecoder(const Output<Node>& input,
                                 const Output<Node>& seq_len,
                                 const bool ctc_merge_repeated);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool get_ctc_merge_repeated() const { return m_ctc_merge_repeated; }

            private:
                bool m_ctc_merge_repeated;
            };
        } // namespace v0
        using v0::CTCGreedyDecoder;
    } // namespace op
} // namespace ngraph
