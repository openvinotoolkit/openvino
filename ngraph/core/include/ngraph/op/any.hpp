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
            /// \brief Logical "any" reduction operation.
            class NGRAPH_DEPRECATED(
                "This operation is deprecated and will be removed soon. Please don't use it.")
                NGRAPH_API Any : public util::LogicalReduction
            {
                NGRAPH_SUPPRESS_DEPRECATED_START
            public:
                static constexpr NodeTypeInfo type_info{"Any", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs an "any" reduction operation.
                Any() = default;
                /// \brief Constructs an "any" reduction operation.
                ///
                /// \param arg The tensor to be reduced.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                Any(const Output<Node>& arg, const AxisSet& reduction_axes);
                /// \brief Constructs an "any" reduction operation.
                ///
                /// \param arg The tensor to be reduced.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                Any(const Output<Node>& arg, const Output<Node>& reduction_axes);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool visit_attributes(AttributeVisitor& visitor) override { return true; }
                /// \return The default value for Any.
                virtual std::shared_ptr<Node> get_default_value() const override;
                NGRAPH_SUPPRESS_DEPRECATED_END
            };
        }
        NGRAPH_SUPPRESS_DEPRECATED_START
        using v0::Any;
        NGRAPH_SUPPRESS_DEPRECATED_END
    }
}
