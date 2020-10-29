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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v4
        {
            /// \brief A Swish Activation Function
            /// f(x) =  x / (1.0 + exp(-beta * x)) or
            /// f(x) = x * sigmoid(beta * x)
            ///
            class NGRAPH_API Swish : public ngraph::op::Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Swish", 4};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Swish() = default;

                /// \brief Constructs an Swish operation.
                ///
                /// \param data Input tensor
                /// \param beta Scalar with beta value. If the argument is not specified then use
                /// the default value 1.0
                Swish(const Output<Node>& arg, const Output<Node>& beta);
                explicit Swish(const Output<Node>& arg);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
            };
        }
    }
}
