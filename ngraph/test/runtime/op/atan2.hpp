// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "backend_visibility.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Elementwise full arctan operation
            class BACKEND_API Atan2 : public util::BinaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Atan2", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Atan2()
                    : util::BinaryElementwiseArithmetic(AutoBroadcastSpec::NONE)
                {
                }

                /// \brief atan2(y,x) is the angle from the origin to the point (x,y) (note reversed
                /// order).
                ///
                /// \param y
                /// \param x
                Atan2(const Output<Node>& y,
                      const Output<Node>& x,
                      const AutoBroadcastSpec& autob = AutoBroadcastSpec());
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool visit_attributes(AttributeVisitor& visitor) override;
            };
        }
    }
}
