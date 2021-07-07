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
            /// \brief Performs a clipping operation on all elements of the input node
            ///
            /// All input values that are outside of the <min;max> range are set to 'min' or 'max'
            /// depending on which side of the <min;max> range they are. The values that fall into
            /// this range remain unchanged.
            class NGRAPH_API Clamp : public ngraph::op::Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                Clamp();
                /// \brief Constructs a Clamp node.
                ///
                /// \param data - Node producing the input tensor
                /// \param min - the lower bound of the <min;max> range
                /// \param max - the upper bound of the <min;max> range
                Clamp(const Output<Node>& data, const double min, const double max);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool visit_attributes(AttributeVisitor& visitor) override;

                double get_min() const { return m_min; }
                double get_max() const { return m_max; }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            private:
                double m_min;
                double m_max;
            };
        } // namespace v0
        using v0::Clamp;
    } // namespace op
} // namespace ngraph
