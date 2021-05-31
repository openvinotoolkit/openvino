// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Elementwise division operation.
            class NGRAPH_API Divide : public util::BinaryElementwiseArithmetic
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                /// \brief Constructs a division operation.
                Divide()
                    : util::BinaryElementwiseArithmetic(AutoBroadcastSpec::NUMPY)
                {
                }

                /// \brief Constructs a division operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param pythondiv Use Python style rounding for integral type
                /// \param auto_broadcast Auto broadcast specification
                Divide(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       bool pythondiv,
                       const AutoBroadcastSpec& auto_broadcast =
                           AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                /// \brief Constructs a division operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param auto_broadcast Auto broadcast specification
                Divide(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       const AutoBroadcastSpec& auto_broadcast =
                           AutoBroadcastSpec(AutoBroadcastType::NUMPY));
                bool visit_attributes(AttributeVisitor& visitor) override;
                bool is_pythondiv() const { return m_pythondiv; }
                void set_is_pythondiv(bool pythondiv) { m_pythondiv = pythondiv; }
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                size_t get_version() const override { return 1; }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            protected:
                bool m_pythondiv{true};
            };
        } // namespace v1
    }     // namespace op
} // namespace ngraph
