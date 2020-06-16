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
            /// \brief Elementwise cosine operation.
            class NGRAPH_API Cos : public util::UnaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Cos", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a cosine operation.
                Cos() = default;
                /// \brief Constructs a cosine operation.
                ///
                /// \param arg Node that produces the input tensor.
                Cos(const Output<Node>& arg);
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
#ifdef NGRAPH_EVALUATE_ENABLE
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;
#endif

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
            };
        }
        using v0::Cos;
    }
}
