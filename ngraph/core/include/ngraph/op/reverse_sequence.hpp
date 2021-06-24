// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API ReverseSequence : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ReverseSequence", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ReverseSequence() = default;
                /// \brief Constructs a ReverseSequence operation.
                ///
                /// \param arg         tensor with input data to reverse
                /// \param seq_lengths 1D tensor of integers with sequence lengths in the input
                /// tensor.
                /// \param batch_axis  index of the batch dimension.
                /// \param seq_axis    index of the sequence dimension.
                ReverseSequence(const Output<Node>& arg,
                                const Output<Node>& seq_lengths,
                                int64_t batch_axis,
                                int64_t seq_axis);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                size_t get_batch_axis() const { return m_normalized_batch_axis; }
                int64_t get_origin_batch_axis() const { return m_batch_axis; }
                void set_batch_axis(int64_t batch_axis) { m_batch_axis = batch_axis; }
                size_t get_sequence_axis() const { return m_normalized_seq_axis; }
                int64_t get_origin_sequence_axis() const { return m_seq_axis; }
                void set_sequence_axis(int64_t sequence_axis) { m_seq_axis = sequence_axis; }

            private:
                int64_t m_batch_axis;
                int64_t m_seq_axis = 1;
                size_t m_normalized_batch_axis;
                size_t m_normalized_seq_axis;
            };
        } // namespace v0
        using v0::ReverseSequence;
    } // namespace op
} // namespace ngraph
