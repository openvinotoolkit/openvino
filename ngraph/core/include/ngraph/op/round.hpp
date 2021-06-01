// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v5
        {
            /// \brief Elementwise round operation. The output is round to the nearest integer
            /// for each value. In case of halfs, the rule is defined in attribute 'mode':
            ///     'HALF_TO_EVEN' - round halfs to the nearest even integer.
            ///     'HALF_AWAY_FROM_ZERO': - round in such a way that the result heads away from
            /// zero.

            class NGRAPH_API Round : public ngraph::op::Op
            {
            public:
                enum class RoundMode
                {
                    HALF_TO_EVEN,
                    HALF_AWAY_FROM_ZERO
                };
                NGRAPH_RTTI_DECLARATION;

                /// \brief Constructs a round operation.
                Round() = default;

                /// \brief Constructs a round operation.
                ///
                /// \param arg Node that produces the input tensor.
                /// \param mode Rule to resolve halfs
                Round(const Output<Node>& arg, const RoundMode mode);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

                RoundMode get_mode() const { return m_mode; }

            private:
                RoundMode m_mode;
            };
        } // namespace v5
    }     // namespace op
    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const op::v5::Round::RoundMode& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v5::Round::RoundMode>
        : public EnumAttributeAdapterBase<op::v5::Round::RoundMode>
    {
    public:
        AttributeAdapter(op::v5::Round::RoundMode& value)
            : EnumAttributeAdapterBase<op::v5::Round::RoundMode>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::v5::Round::RoundMode>",
                                                    5};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
} // namespace ngraph
