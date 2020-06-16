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
            /// \brief Elementwise natural exponential (exp) operation.
            class NGRAPH_API Exp : public util::UnaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Exp", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs an exponential operation.
                Exp() = default;
                /// \brief Constructs an exponential operation.
                ///
                /// \param arg Node that produces the input tensor.
                Exp(const Output<Node>& arg);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
#ifdef NGRAPH_EVALUATE_ENABLE
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;
#endif
            };
        }
        using v0::Exp;
    }
}
