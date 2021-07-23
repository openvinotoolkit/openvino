// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/util/multi_subgraph_base.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v8
        {
            /// \brief  If operation.
            class NGRAPH_API If : public util::MultiSubGraphOp
            {
            public:
                enum BodyIndexes
                {
                    then_body_index = 0,
                    else_body_index = 1
                };

                NGRAPH_RTTI_DECLARATION;
                bool visit_attributes(AttributeVisitor& visitor) override;

                /// \brief     Constructs If with condition
                ///
                /// \param     execution_condition   condition node.
                If(const Output<Node>& execution_condition);
                If();
                explicit If(const OutputVector& values);

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return then_body as ngraph::Function.
                std::shared_ptr<Function> get_then_body() const
                {
                    return m_bodies[then_body_index];
                }
                /// \return else_body as ngraph::Function.
                std::shared_ptr<Function> get_else_body() const
                {
                    return m_bodies[else_body_index];
                }
                /// \brief     sets new ngraph::Function as new then_body.
                ///
                /// \param     body   new body for 'then' branch.
                void set_then_body(const std::shared_ptr<Function>& body)
                {
                    m_bodies[then_body_index] = body;
                }
                /// \brief     sets new ngraph::Function as new else_body.
                ///
                /// \param     body   new body for 'else' branch.
                void set_else_body(const std::shared_ptr<Function>& body)
                {
                    m_bodies[else_body_index] = body;
                }
                /// \brief     sets new input to the operation associated with parameters
                /// of each sub-graphs
                ///
                /// \param     value           input to operation
                /// \param     then_parameter  parameter for then_body
                /// \param     else_parameter  parameter for else_body
                void set_input(const Output<Node>& value,
                               const std::shared_ptr<Parameter> then_parameter,
                               const std::shared_ptr<Parameter> else_parameter);
                /// \brief     sets new output from the operation associated with results
                /// of each sub-graphs
                ///
                /// \param     then_result     result from then_body
                /// \param     else_parameter  pesult from else_body
                /// \return    output from operation
                Output<Node> set_output(const std::shared_ptr<Result> then_result,
                                        const std::shared_ptr<Result> else_result);

                void validate_and_infer_types() override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;

            private:
                void validate_and_infer_type_body(
                    std::shared_ptr<Function> body,
                    ngraph::op::util::MultiSubgraphInputDescriptionVector& input_descriptors);
                void fill_body(std::shared_ptr<op::v8::If> new_op,
                               size_t branch_index,
                               const OutputVector& new_args) const;
            };
        } // namespace v8
    }     // namespace op
} // namespace ngraph