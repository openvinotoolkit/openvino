// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Exponential Linear Unit
            /// x <  0 => f(x) = alpha * (exp(x) - 1.)
            /// x >= 0 => f(x) = x
            ///
            class NGRAPH_API Elu : public ngraph::op::Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Elu", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Elu() = default;
                /// \brief Constructs an Elu operation.
                ///
                /// \param data Input tensor
                /// \param alpha Multiplier for negative values
                Elu(const Output<Node>& data, const double alpha);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                double get_alpha() const { return m_alpha; }

            private:
                double m_alpha;
            };
        } // namespace v0
        using v0::Elu;
    } // namespace op
} // namespace ngraph
