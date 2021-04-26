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
            /// \brief Mod returns an element-wise division reminder with two given tensors applying
            /// multi-directional broadcast rules.
            class NGRAPH_API Mod : public util::BinaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Mod", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a Mod node.
                Mod()
                    : util::BinaryElementwiseArithmetic(AutoBroadcastSpec::NUMPY)
                {
                }
                ///
                /// \param A - Dividend tensor
                /// \param B - Divisor tensor
                /// \param auto_broadcast Auto broadcast specification
                Mod(const Output<Node>& A,
                    const Output<Node>& B,
                    const AutoBroadcastSpec& auto_broadcast =
                        AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v1
    }     // namespace op
} // namespace ngraph
