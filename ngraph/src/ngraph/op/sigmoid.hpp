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

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API Sigmoid : public util::UnaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Sigmoid", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Sigmoid(const Output<Node>& arg);
                Sigmoid() = default;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;
            };

            /// \brief Elementwise SigmoidBackprop operation.
            ///
            class NGRAPH_API SigmoidBackprop : public util::BinaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"SigmoidBackprop", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                SigmoidBackprop()
                    : util::BinaryElementwiseArithmetic(AutoBroadcastSpec::NONE)
                {
                }

                /// \brief Constructs a SigmoidBackprop operation.
                ///
                /// \param arg Node that produces the Sigmoid forward input tensor.
                SigmoidBackprop(const Output<Node>& arg, const Output<Node>& delta);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }
        using v0::Sigmoid;
        using v0::SigmoidBackprop;
    }
}
