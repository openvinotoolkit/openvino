// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/variable.hpp"
#include "ngraph/op/util/variable_extension.hpp"

namespace ngraph
{
    namespace op
    {
        class NGRAPH_API ReadValueBase : public Op, public VariableExtension
        {
        public:
            NGRAPH_RTTI_DECLARATION;

            ReadValueBase() = default;

            /// \brief Constructs an AssignBase operation.
            explicit ReadValueBase(const OutputVector& arguments)
                : Op(arguments)
            {
            }
        };

        namespace v3
        {
            /// \brief ReadValue operation creates the variable with `variable_id` and returns value
            /// of this variable.
            class NGRAPH_API ReadValue : public ReadValueBase
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                ReadValue() = default;

                /// \brief Constructs a ReadValue operation.
                ///
                /// \param init_value   Node that produces the input tensor.
                /// \param variable_id  identificator of the variable to create.
                ReadValue(const Output<Node>& init_value, const std::string& variable_id);

                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool visit_attributes(AttributeVisitor& visitor) override;

                std::string get_variable_id() const override { return m_variable_id; }

            private:
                std::string m_variable_id;
            };
        } // namespace v3

        namespace v6
        {
            /// \brief ReadValue operation gets an input value from the variable with `variable_id`
            /// and returns it as an output.
            class NGRAPH_API ReadValue : public ReadValueBase
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                ReadValue() = default;

                /// \brief Constructs a ReadValue operation.
                ///
                /// \param init_value Node that produces the input tensor.
                /// \param variable Class for storing and synchronizing element types, shapes and
                /// identifiers
                /// between pairs of Assign/ReadValue nodes.
                ReadValue(const Output<Node>& init_value,
                          const std::shared_ptr<Variable>& variable);

                void validate_and_infer_types() override;

                void revalidate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool visit_attributes(AttributeVisitor& visitor) override;

                std::string get_variable_id() const override
                {
                    NGRAPH_CHECK(m_variable,
                                 "Variable is not initialized. Variable_id is unavailable");
                    return m_variable->get_info().variable_id;
                }

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs,
                              const EvaluationContext& evaluation_context) const override;
                bool has_evaluate() const override;

                bool constant_fold(OutputVector& output_values,
                                   const OutputVector& inputs_values) override;
            };
        } // namespace v6
    }     // namespace op
} // namespace ngraph
