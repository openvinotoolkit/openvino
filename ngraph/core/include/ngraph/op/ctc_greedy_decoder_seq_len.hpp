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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v6
        {
            class NGRAPH_API CTCGreedyDecoderSeqLen : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                //static constexpr NodeTypeInfo type_info{"CTCGreedyDecoderSeqLen", 0};
                //const NodeTypeInfo& get_type_info() const override { return type_info; }
                CTCGreedyDecoderSeqLen() = default;
                /// \brief Constructs a CTCGreedyDecoderSeqLen operation
                ///
                /// \param input              Logits on which greedy decoding is performed
                /// \param seq_len            Sequence lengths
                /// \param merge_repeated Whether to merge repeated labels
                CTCGreedyDecoderSeqLen(const Output<Node>& input,
                                       const Output<Node>& seq_len,
                                       const bool merge_repeated,
                                       const element::Type& classes_index_type = element::i32,
                                       const element::Type& sequence_length_type = element::i32);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool get_merge_repeated() const { return m_merge_repeated; }

                element::Type get_classes_index_type() const { return m_classes_index_type; }
                void set_classes_index_type(const element::Type& classes_index_type)
                {
                    m_classes_index_type = classes_index_type;
                }

                element::Type get_sequence_length_type() const { return m_sequence_length_type; }
                void set_sequence_length_type(const element::Type& sequence_length_type)
                {
                    m_sequence_length_type = sequence_length_type;
                }
            private:
                bool m_merge_repeated;
                element::Type m_classes_index_type{element::i32};
                element::Type m_sequence_length_type{element::i32};
            };
        } // namespace v6
        using v6::CTCGreedyDecoderSeqLen;
    } // namespace op
} // namespace ngraph
