// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Elementwise FloorMod operation.
            ///
            class NGRAPH_API FloorMod : public util::BinaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"FloorMod", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs an uninitialized addition operation
                FloorMod()
                    : util::BinaryElementwiseArithmetic(AutoBroadcastSpec::NUMPY){};

                /// \brief Constructs an Floor Mod operation.
                ///
                /// \param arg0 Output that produces the first input tensor.<br>
                /// `[d0, ...]`
                /// \param arg1 Output that produces the second input tensor.<br>
                /// `[d0, ...]`
                /// \param auto_broadcast Auto broadcast specification
                ///
                /// Output `[d0, ...]`
                ///
                FloorMod(const Output<Node>& arg0,
                         const Output<Node>& arg1,
                         const AutoBroadcastSpec& auto_broadcast = AutoBroadcastType::NUMPY);

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;
            };
        } // namespace v1

        using v1::FloorMod;
    } // namespace op
} // namespace ngraph
