//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/op/util/logical_reduction.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Logical "all" reduction operation.
            class NGRAPH_API All : public util::LogicalReduction
            {
            public:
                static constexpr NodeTypeInfo type_info{"All", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs an "all" reduction operation.
                All() = default;
                /// \brief Constructs an "all" reduction operation.
                ///
                /// \param arg The tensor to be reduced.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                All(const Output<Node>& arg, const AxisSet& reduction_axes);
                /// \brief Constructs an "all" reduction operation.
                ///
                /// \param arg The tensor to be reduced.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                All(const Output<Node>& arg, const Output<Node>& reduction_axes);
                bool visit_attributes(AttributeVisitor& visitor) override { return true; }
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The default value for All.
                virtual std::shared_ptr<Node> get_default_value() const override;
            };
        }
        using v0::All;
    }
}
