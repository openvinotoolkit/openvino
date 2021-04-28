// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    class Function;
    namespace op
    {
        namespace v0
        {
            /// \brief A function parameter.
            ///
            /// Parameters are nodes that represent the arguments that will be passed to
            /// user-defined functions. Function creation requires a sequence of parameters.
            /// Basic graph operations do not need parameters attached to a function.
            class NGRAPH_API Parameter : public op::Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Parameter", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructions a tensor-typed parameter node.
                Parameter() = default;
                /// \brief Constructions a tensor-typed parameter node.
                ///
                /// \param element_type The element type of the parameter.
                /// \param pshape The partial shape of the parameter.
                Parameter(const ngraph::element::Type& element_type, const PartialShape& pshape);

                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool is_relevant_to_shapes() const;
                void set_is_relevant_to_shapes(bool is_relevant);

                const PartialShape& get_partial_shape() const { return m_partial_shape; }
                PartialShape& get_partial_shape() { return m_partial_shape; }
                void set_partial_shape(const PartialShape& partial_shape)
                {
                    m_partial_shape = partial_shape;
                }
                const element::Type& get_element_type() const { return m_element_type; }
                void set_element_type(const element::Type& element_type)
                {
                    m_element_type = element_type;
                }

            protected:
                PartialShape m_partial_shape;
                element::Type m_element_type;
                bool m_is_relevant_to_shapes;
            };
        } // namespace v0
        using v0::Parameter;
    } // namespace op
    using ParameterVector = std::vector<std::shared_ptr<op::Parameter>>;

    template <>
    class NGRAPH_API AttributeAdapter<ParameterVector> : public VisitorAdapter
    {
    public:
        AttributeAdapter(ParameterVector& ref);

        bool visit_attributes(AttributeVisitor& visitor) override;

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<ParameterVector>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }

    protected:
        ParameterVector& m_ref;
    };
} // namespace ngraph
