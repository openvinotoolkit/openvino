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
#include "ngraph/op/util/variable.hpp"

namespace ngraph
{
    namespace op
    {
        class NGRAPH_API ReadValueBase : public Op
        {
        public:
            NGRAPH_RTTI_DECLARATION;

            ReadValueBase() = default;

            /// \brief Constructs an AssignBase operation.
            explicit ReadValueBase(const OutputVector& arguments)
                : Op(arguments)
            {
            }

            /// \brief Sets the identifier of corresponding variable
            ///
            /// \param variable_id New identifier of the variable.
            virtual void set_variable_id(const std::string& variable_id){};

            /// \brief Returns the identifier of corresponding variable.
            virtual std::string get_variable_id() const = 0;

            /// \brief Returns variable connected to this node.
            virtual std::shared_ptr<ngraph::Variable> get_variable() const { return m_variable; }
            /// \brief Sets a new variable to be connected to this node.
            ///
            /// \param variable New variable to be connected to this node.
            virtual void set_variable(const std::shared_ptr<ngraph::Variable>& variable)
            {
                m_variable = variable;
            }

        protected:
            std::shared_ptr<ngraph::Variable> m_variable;
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
                void set_variable_id(const std::string& variable_id) override
                {
                    m_variable_id = variable_id;
                }

            private:
                std::string m_variable_id;
            };
        }

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

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool visit_attributes(AttributeVisitor& visitor) override;

                std::string get_variable_id() const override
                {
                    NGRAPH_CHECK(m_variable,
                                 "Variable is not initialized. Variable_id is unavailable");
                    return m_variable->get_info().variable_id;
                }
            };
        }
    }
}
