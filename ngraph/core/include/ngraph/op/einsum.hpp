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
        namespace v7
        {
            /// \brief Einsum operation.
            class NGRAPH_API Einsum : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                Einsum() = default;

                ///
                /// \brief      Constructs Einsum operation.
                ///
                /// \param      inputs        Input nodes on which Einsum operation performs
                /// contraction
                ///
                /// \param      equation      Einstein summation convention
                ///
                Einsum(const OutputVector& inputs, const std::string& equation);

                void validate_and_infer_types() override;

                bool visit_attributes(AttributeVisitor& visitor) override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            private:
                std::string m_equation;

                /// \brief      Check equation format and extracts input subscripts and output
                /// subscript
                ///
                /// \param      input_subscripts      A vector of extracted input subscripts
                /// \param      output_subscript      An output subscript
                ///
                bool parse_equation(std::vector<std::string>& input_subscripts,
                                    std::string& output_subscript);

                /// \brief      Extract labels from subscript
                ///
                /// \param      subscript      Input subscript
                /// \param      labels         Extracted labels
                ///
                void extract_labels(std::string const& subscript, std::vector<std::string>& labels);
            };
        } // namespace v7
    }     // namespace op
} // namespace ngraph
