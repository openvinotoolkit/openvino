//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
