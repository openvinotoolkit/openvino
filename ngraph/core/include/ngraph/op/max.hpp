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
#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Max-reduction operation.
            class NGRAPH_DEPRECATED(
                "This operation is deprecated and will be removed soon. "
                "Use v1::ReduceMax instead of it.") NGRAPH_API Max
                : public util::ArithmeticReduction
            {
                NGRAPH_SUPPRESS_DEPRECATED_START
            public:
                static constexpr NodeTypeInfo type_info{"Max", 0};

                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a "max" reduction operation.
                Max() = default;

                /// \brief Constructs a max-reduction operation.
                ///
                /// \param arg The tensor to be reduced.
                /// \param reduction_axes The axis positions (0-based) to be elimaxated.
                Max(const Output<Node>& arg, const AxisSet& reduction_axes);

                /// \brief Constructs a "max" reduction operation.
                ///
                /// \param arg The tensor to be reduced.
                /// \param reduction_axes The axis positions (0-based) to be elimaxated.
                Max(const Output<Node>& arg, const Output<Node>& reduction_axes);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The default value for Max.
                virtual std::shared_ptr<Node> get_default_value() const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                NGRAPH_SUPPRESS_DEPRECATED_END
            };
        }

        namespace v1
        {
            class NGRAPH_API ReduceMax : public util::ArithmeticReductionKeepDims
            {
            public:
                static constexpr NodeTypeInfo type_info{"ReduceMax", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a summation operation.
                ReduceMax() = default;
                /// \brief Constructs a summation operation.
                ///
                /// \param arg The tensor to be summed.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                /// \param keep_dims If set to 1 it holds axes that are used for reduction.
                ReduceMax(const Output<Node>& arg,
                          const Output<Node>& reduction_axes,
                          bool keep_dims = false);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
            };
        }

        NGRAPH_SUPPRESS_DEPRECATED_START
        using v0::Max;
        NGRAPH_SUPPRESS_DEPRECATED_END
    }
}
