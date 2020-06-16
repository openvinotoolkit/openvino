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

#include "ngraph/axis_set.hpp"
#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            // clang-format off
            /// \brief Tensor sum operation.
            ///
            /// Element-wise sums the input tensor, eliminating the specified reduction axes.
            /// For example:
            ///
            /// \f[
            ///     \mathit{sum}\left(\{0\},
            ///         \left[ \begin{array}{ccc}
            ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
            ///     \left[ (1 + 3 + 5), (2 + 4 + 6) \right] =
            ///     \left[ 9, 12 \right]~~~\text{(dimension 0 (rows) is eliminated)}
            /// \f]
            ///
            /// \f[
            ///     \mathit{sum}\left(\{1\},
            ///         \left[ \begin{array}{ccc}
            ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
            ///     \left[ (1 + 2), (3 + 4), (5 + 6) \right] =
            ///     \left[ 3, 7, 11 \right]~~~\text{(dimension 1 (columns) is eliminated)}
            /// \f]
            ///
            /// \f[
            ///     \mathit{sum}\left(\{0,1\},
            ///         \left[ \begin{array}{ccc}
            ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
            ///      (1 + 2) + (3 + 4) + (5 + 6) =
            ///      21~~~\text{(both dimensions (rows and columns) are eliminated)}
            /// \f]
            ///
            /// ## Parameters
            ///
            /// |                      | Description                                            |
            /// | -------------------- | ----------------------------------------               |
            /// | `reduction_axes`     | The axes to eliminate through summation.               |
            /// | `keep_dims`          | If set to 1 it holds axes that are used for reduction. |
            ///
            /// ## Inputs
            ///
            /// |       | Type                              | Description                                            |
            /// | ----- | --------------------------------- | ------------------------------------------------------ |
            /// | `arg` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any shape and numeric element type. |
            ///
            /// ## Output
            ///
            /// | Type                                      | Description                                                                                                      |
            /// | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
            /// | \f$N[\textit{delete}(A,d_1,\dots,d_n)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with the `reduction_axes` \f$A\f$ eliminated by summation. |
            // clang-format off
            class NGRAPH_API ReduceSum : public util::ArithmeticReductionKeepDims
            {
            public:
                static constexpr NodeTypeInfo type_info{"ReduceSum", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a summation operation.
                ReduceSum() = default;
                /// \brief Constructs a summation operation.
                ///
                /// \param arg The tensor to be summed.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                /// \param keep_dims If set to 1 it holds axes that are used for reduction.
                ReduceSum(const Output<Node>& arg,
                          const Output<Node>& reduction_axes,
                          bool keep_dims = false);

                size_t get_version() const override { return 1; }

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The default value for Sum.
                virtual std::shared_ptr<Node> get_default_value() const override;

#ifdef NGRAPH_EVALUATE_ENABLE
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;
#endif

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
            };
        }
    }
}
