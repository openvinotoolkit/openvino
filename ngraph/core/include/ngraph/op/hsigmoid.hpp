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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v5
        {
            /// \brief A HSigmoid Activation Function
            /// f(x) = min(max(x + 3, 0), 6) / 6 or
            /// f(x) = min(ReLU(x + 3), 6) / 6
            ///
            class NGRAPH_API HSigmoid : public ngraph::op::util::UnaryElementwiseArithmetic
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                HSigmoid() = default;

                /// \brief Constructs a HSigmoid operation.
                ///
                /// \param data Input tensor
                HSigmoid(const Output<Node>& arg);

                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
            };
        }
    }
}
