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

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Operator performing Matrix Multiplication.
            class NGRAPH_API MatMul : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                MatMul() = default;
                /// \brief Constructs an Matrix Multiplication operation.
                ///
                /// \param A Matrix A
                /// \param B Matrix B
                /// \param transpose_a If matrix A should be transposed.
                /// \param transpose_b If matrix B should be transposed.
                MatMul(const Output<Node>& A,
                       const Output<Node>& B,
                       const bool& transpose_a = 0,
                       const bool& transpose_b = 0);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;

                bool get_transpose_a() const { return m_transpose_a; }
                bool get_transpose_b() const { return m_transpose_b; }
                void set_transpose_a(bool transpose_a) { m_transpose_a = transpose_a; }
                void set_transpose_b(bool transpose_b) { m_transpose_b = transpose_b; }

            private:
                bool m_transpose_a;
                bool m_transpose_b;
            };
        }
        using v0::MatMul;
    } // namespace op
} // namespace ngraph
