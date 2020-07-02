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

#include "backend_visibility.hpp"
#include "ngraph/op/util/index_reduction.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Computes minimum index along a specified axis for a given tensor
            class BACKEND_API ArgMax : public op::util::IndexReduction
            {
            public:
                static constexpr NodeTypeInfo type_info{"ArgMax", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a ArgMax operation.
                ArgMax() = default;
                /// \brief Constructs a ArgMax operation.
                ///
                /// \param arg The input tensor
                /// \param axis The axis along which to compute an index for maximum
                /// \param index_element_type produce indices. Currently, only int64 or int32 are
                ///                           supported
                ArgMax(const Output<Node>& arg,
                       size_t axis,
                       const element::Type& index_element_type);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node> get_default_value() const override;
            };
        }
    }
}
