// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Tensor transpose operation.
            class NGRAPH_API Transpose : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                Transpose() = default;
                ///
                /// \brief      Constructs a transpose operation.
                ///
                /// \param      arg          Node producing the tensor to be transposed.
                /// \param      input_order  Node producing the permutation to apply to the axes
                ///                          of the input shape. Must be a vector with shape [n],
                ///                          where n is the rank of arg. The tensor's value must
                ///                          contain every integer in the range [0, n-1].
                ///
                Transpose(const Output<Node>& arg, const Output<Node>& input_order);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v1
        using v1::Transpose;
    } // namespace op
} // namespace ngraph
