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
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Operator performing Stack.
            class NGRAPH_API Stack : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"Stack", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Stack() = default;

                /// \brief Constructs a stack operation.
                ///
                /// \param args The outputs producing the input tensors.
                /// \param axis The axis in the result array along which the input arrays are
                /// stacked.
                Stack(const OutputVector& args, int64_t axis);

                /// \brief Constructs a stack operation.
                ///
                /// \param args The nodes producing the input tensors.
                /// \param axis The axis in the result array along which the input arrays are
                /// stacked.
                Stack(const NodeVector& args, int64_t axis);

                void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const OutputVector& deltas) override;

                virtual void pre_validate_and_infer_types() override;

                virtual NodeVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The stack axis
                int64_t get_axis() const { return m_axis; }
                void set_axis(int64_t axis) { m_axis = axis; }
            private:
                int64_t m_axis;
            };
        }
        using v0::Stack;
    } // namespace op
} // namespace ngraph
