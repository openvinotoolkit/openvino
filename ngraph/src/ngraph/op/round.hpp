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

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Elementwise round operation.
            class NGRAPH_API Round : public util::UnaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Round", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a round operation.
                Round() = default;

                /// \brief Constructs a round operation. The output is round to the nearest integer
                /// for each value. In case of halfs, the rule is to round them to the nearest even
                /// integer.
                ///
                /// \param arg Node that produces the input tensor.
                Round(const Output<Node>& arg);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;
            };
        }
        using v0::Round;
    }
}
