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

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Matrix multiply for a batch of Rank 2 tensors.
            /// The inputs are expected to be Rank 3, where the first dim is the
            /// batch size and must be the same for both inputs. The last two dims
            /// are the shape of matrices, i.e. `(batch_size, :, :)`.
            /// For example, for `a` with shape `(batch_size, n, k)`, and `b` with
            /// shape `(batch_size, k, m)`, the result of BatchMatMul will have shape
            /// `(batch_size, n, m)`, and `BatchMatMul(a, b)[i] = Dot(a[i], b[i])`.
            class NGRAPH_API BatchMatMul : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"BatchMatMul", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                BatchMatMul() = default;
                /// \brief Constructs a batch of matmul product operation.
                ///
                /// \param arg0 The node producing the first argument.
                /// \param arg1 The node producing the second argument.
                BatchMatMul(const Output<Node>& arg0, const Output<Node>& arg1);

                virtual void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
            };
        }
        using v0::BatchMatMul;

        namespace util
        {
            std::shared_ptr<Node> batch_mat_transpose(const std::shared_ptr<Node>& node);
        }
    }
}
