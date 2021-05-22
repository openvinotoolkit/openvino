// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Calculates an element-wise squared difference between two tensors
            ///
            /// y[i] = (x1[i] - x2[i])^2
            class NGRAPH_API SquaredDifference : public util::BinaryElementwiseArithmetic
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Constrcuts an uninitialized squared difference operation
                SquaredDifference()
                    : util::BinaryElementwiseArithmetic(AutoBroadcastSpec::NUMPY)
                {
                }
                /// \brief Constructs the squared difference operation.
                ///
                /// \param x1 First input tensor
                /// \param x2 Second input tensor
                /// \param auto_broadcast Auto broadcast specification
                SquaredDifference(const Output<Node>& x1,
                                  const Output<Node>& x2,
                                  const AutoBroadcastSpec& auto_broadcast =
                                      AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v0
        using v0::SquaredDifference;
    } // namespace op
} // namespace ngraph
