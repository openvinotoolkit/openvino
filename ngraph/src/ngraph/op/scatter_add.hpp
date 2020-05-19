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
#include "ngraph/op/util/scatter_base.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Add updates to slices from inputs addressed by indices
            class NGRAPH_API ScatterAdd : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ScatterAdd", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ScatterAdd() = default;
                /// \param inputs Tensor
                /// \param indices Index tensor: Data type must be `element::i32` or `element::i64`
                /// \param updates Tensor: Must have same type as inputs
                ScatterAdd(const Output<Node>& inputs,
                           const Output<Node>& indices,
                           const Output<Node>& updates)
                    : Op({inputs, indices, updates})
                {
                    constructor_validate_and_infer_types();
                }

                void validate_and_infer_types() override;

                void generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                       const OutputVector& /* deltas */) override
                {
                    throw ngraph_error("Not yet implemented");
                }
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }

        namespace v3
        {
            ///
            /// \brief      Add updates to slices from inputs addressed by indices
            ///
            class NGRAPH_API ScatterAdd : public util::ScatterBase
            {
            public:
                static constexpr NodeTypeInfo type_info{"ScatterAdd", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ScatterAdd() = default;

                ///
                /// \brief      Constructs ScatterAdd object.
                ///
                /// \param      data     The input tensor to be updated.
                /// \param      indices  The tensor with indexes which will be updated.
                /// \param      updates  The tensor with update values.
                /// \param[in]  axis     The axis at which elements will be updated.
                ///
                ScatterAdd(const Output<Node>& data,
                           const Output<Node>& indices,
                           const Output<Node>& updates,
                           const Output<Node>& axis);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& inputs) const override;
            };
        }
        using v0::ScatterAdd;
    }
}
