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

#include "ngraph/op/util/arithmetic_reduction.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Product reduction operation.
            ///
            /// Reduces the tensor, eliminating the specified reduction axes by taking the product.
            class NGRAPH_API Product : public util::ArithmeticReduction
            {
            public:
                static constexpr NodeTypeInfo type_info{"Product", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a product reduction operation.
                Product() = default;
                /// \brief Constructs a product reduction operation.
                ///
                /// \param arg The tensor to be reduced.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                Product(const Output<Node>& arg, const AxisSet& reduction_axes);
                /// \brief Constructs a product reduction operation.
                ///
                /// \param arg The tensor to be reduced.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                Product(const Output<Node>& arg, const Output<Node>& reduction_axes);

                /// \return The default value for Product.
                virtual std::shared_ptr<Node> get_default_value() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

#ifdef NGRAPH_EVALUATE_ENABLE
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;
#endif
            };
        }
        // default opset version
        using v0::Product;
    }
}
