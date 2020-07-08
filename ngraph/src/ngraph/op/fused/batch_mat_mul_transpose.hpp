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
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Matrix multiply for a batch of Rank 2 tensors each with potential
            /// transpose.
            ///
            /// The inputs are expected to be Rank 3, where the first dim is the
            /// batch size and must be the same for both inputs. The last two dims
            /// are the shape of matrices, i.e. `(batch_size, :, :)`.
            /// For example, for `a` with shape `(batch_size, n, k)`, and `b` with
            /// shape `(batch_size, k, m)`, the result of BatchMatMul will have shape
            /// `(batch_size, n, m)`, and `BatchMatMulTranspose(a, b)[i] = Dot(a[i], b[i])`.
            class NGRAPH_API BatchMatMulTranspose : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"BatchMatMulTranspose", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                BatchMatMulTranspose() = default;
                /// \brief Constructs a batch of matmul product operation.
                ///
                /// \param arg0 The node producing the first argument.
                /// \param arg1 The node producing the second argument.
                /// \param transpose_0 Apply transpose to arg0.
                /// \param transpose_1 Apply transpose to arg1.
                BatchMatMulTranspose(const Output<Node>& arg0,
                                     const Output<Node>& arg1,
                                     bool transpose_0 = false,
                                     bool transpose_1 = false);

                bool get_transpose_arg0() const { return m_transpose_arg0; }
                bool get_transpose_arg1() const { return m_transpose_arg1; }
                virtual void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual OutputVector decompose_op() const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

            private:
                bool m_transpose_arg0;
                bool m_transpose_arg1;
            };
        }
        using v0::BatchMatMulTranspose;
    }
}
