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
            /// \brief Parametrized Relu
            /// x <  0 => f(x) = x * slope
            /// x >= 0 => f(x) = x
            ///
            class NGRAPH_API PRelu : public ngraph::op::Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                PRelu(const AutoBroadcastSpec& auto_broadcast =
                          AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                /// \brief Constructs a PRelu operation.
                ///
                /// \param data Input tensor
                /// \param slope Multipliers for negative values
                /// \param auto_broadcast Auto broadcast specification. Default is Numpy-style
                ///                       implicit broadcasting.
                PRelu(const Output<Node>& data,
                      const Output<Node>& slope,
                      const AutoBroadcastSpec& auto_broadcast =
                          AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                const AutoBroadcastSpec& get_autob() const override { return m_autob; }
                void set_autob(const AutoBroadcastSpec& autob) { m_autob = autob; }
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                void validate_and_infer_types() override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;

            private:
                AutoBroadcastSpec m_autob;
            };
        }
        using v0::PRelu;
    }
}
