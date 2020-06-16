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
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

#include <memory>

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Elementwise Relu operation.
            ///
            class NGRAPH_API Relu : public ngraph::op::util::UnaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Relu", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Relu() = default;
                /// \brief Constructs a Relu operation.
                ///
                /// \param arg Node that produces the input tensor.
                Relu(const Output<ngraph::Node>& arg);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

#ifdef NGRAPH_EVALUATE_ENABLE
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;
#endif

                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
            };

            /// \brief Elementwise ReluBackprop operation.
            ///
            class NGRAPH_API ReluBackprop : public ngraph::op::util::BinaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"ReluBackprop", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ReluBackprop()
                    : BinaryElementwiseArithmetic(AutoBroadcastSpec::NONE)
                {
                }
                /// \brief Constructs a ReluBackprop operation.
                ///
                /// \param arg Node that produces the relu forward input tensor.
                ReluBackprop(const Output<ngraph::Node>& arg, const Output<ngraph::Node>& delta);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }
        using v0::Relu;
        using v0::ReluBackprop;
    }
}
