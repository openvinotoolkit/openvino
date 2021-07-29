// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Elementwise logical negation operation.
            class NGRAPH_API LogicalNot : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                /// \brief Constructs a logical negation operation.
                LogicalNot() = default;
                /// \brief Constructs a logical negation operation.
                ///
                /// \param arg Node that produces the input tensor.
                LogicalNot(const Output<Node>& arg);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v1
    }     // namespace op
} // namespace ngraph
