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
#include "ngraph/op/util/scatter_nd_base.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Add updates to slices from inputs addressed by indices
            class NGRAPH_API ScatterNDAdd : public util::ScatterNDBase
            {
            public:
                static constexpr NodeTypeInfo type_info{"ScatterNDAdd", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ScatterNDAdd() = default;
                /// \param inputs Tensor
                /// \param indices Index tensor: Data type must be `element::i32` or `element::i64`
                /// \param updates Tensor: Must have same type as inputs
                ScatterNDAdd(const Output<Node>& inputs,
                             const Output<Node>& indices,
                             const Output<Node>& updates)
                    : util::ScatterNDBase(inputs, indices, updates)
                {
                }

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }
        using v0::ScatterNDAdd;
    }
}
