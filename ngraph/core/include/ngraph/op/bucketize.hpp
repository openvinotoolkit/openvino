// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v3
        {
            /// \brief Operation that bucketizes the input based on boundaries
            class NGRAPH_API Bucketize : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Bucketize", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Bucketize() = default;
                /// \brief Constructs a Bucketize node

                /// \param data              Input data to bucketize
                /// \param buckets           1-D of sorted unique boundaries for buckets
                /// \param output_type       Output tensor type, "i64" or "i32", defaults to i64
                /// \param with_right_bound  indicates whether bucket includes the right or left
                ///                          edge of interval. default true = includes right edge
                Bucketize(const Output<Node>& data,
                          const Output<Node>& buckets,
                          const element::Type output_type = element::i64,
                          const bool with_right_bound = true);

                virtual void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& inputs) const override;

                element::Type get_output_type() const { return m_output_type; }
                void set_output_type(element::Type output_type) { m_output_type = output_type; }
                // Overload collision with method on Node
                using Node::set_output_type;

                bool get_with_right_bound() const { return m_with_right_bound; }
                void set_with_right_bound(bool with_right_bound)
                {
                    m_with_right_bound = with_right_bound;
                }

            private:
                element::Type m_output_type;
                bool m_with_right_bound;
            };
        } // namespace v3
        using v3::Bucketize;
    } // namespace op
} // namespace ngraph
