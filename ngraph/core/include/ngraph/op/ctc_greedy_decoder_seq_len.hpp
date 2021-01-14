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
        namespace v6
        {
            /// \brief Operator performing CTCGreedyDecoder
            ///
            class NGRAPH_API CTCGreedyDecoderSeqLen : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                CTCGreedyDecoderSeqLen() = default;
                /// \brief Constructs a CTCGreedyDecoderSeqLen operation
                ///
                /// \param input                3-D tensor of logits on which greedy decoding is
                /// performed
                /// \param seq_len              1-D tensor of sequence lengths
                /// \param merge_repeated       Whether to merge repeated labels
                /// \param classes_index_type   Specifies the output classes_index tensor type
                /// \param sequence_length_type Specifies the output sequence_length tensor type
                CTCGreedyDecoderSeqLen(const Output<Node>& input,
                                       const Output<Node>& seq_len,
                                       const bool merge_repeated = true,
                                       const element::Type& classes_index_type = element::i32,
                                       const element::Type& sequence_length_type = element::i32);
                /// \brief Constructs a CTCGreedyDecoderSeqLen operation
                ///
                /// \param input                3-D tensor of logits on which greedy decoding is
                /// performed
                /// \param seq_len              1-D tensor of sequence lengths
                /// \param blank_index          Scalar or 1-D tensor with 1 element used to mark a
                /// blank index
                /// \param merge_repeated       Whether to merge repeated labels
                /// \param classes_index_type   Specifies the output classes_index tensor type
                /// \param sequence_length_type Specifies the output sequence_length tensor type
                CTCGreedyDecoderSeqLen(const Output<Node>& input,
                                       const Output<Node>& seq_len,
                                       const Output<Node>& blank_index,
                                       const bool merge_repeated = true,
                                       const element::Type& classes_index_type = element::i32,
                                       const element::Type& sequence_length_type = element::i32);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \brief Get merge_repeated attribute
                ///
                /// \return Current value of merge_repeated attribute
                ///
                bool get_merge_repeated() const { return m_merge_repeated; }
                /// \brief Get classes_index_type attribute
                ///
                /// \return Current value of classes_index_type attribute
                ///
                const element::Type& get_classes_index_type() const { return m_classes_index_type; }
                /// \brief Set classes_index_type attribute
                ///
                /// \param classes_index_type Type of classes_index
                ///
                void set_classes_index_type(const element::Type& classes_index_type)
                {
                    m_classes_index_type = classes_index_type;
                    validate_and_infer_types();
                }

                /// \brief Get sequence_length_type attribute
                ///
                /// \return Current value of sequence_length_type attribute
                ///
                const element::Type& get_sequence_length_type() const
                {
                    return m_sequence_length_type;
                }

                /// \brief Set sequence_length_type attribute
                ///
                /// \param sequence_length_type Type of sequence length
                ///
                void set_sequence_length_type(const element::Type& sequence_length_type)
                {
                    m_sequence_length_type = sequence_length_type;
                    validate_and_infer_types();
                }

            private:
                bool m_merge_repeated;
                element::Type m_classes_index_type{element::i32};
                element::Type m_sequence_length_type{element::i32};
            };
        } // namespace v6
    }     // namespace op
} // namespace ngraph
