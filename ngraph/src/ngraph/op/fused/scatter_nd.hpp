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

#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief  Replace values within provided tensor by `updates` according to `indices`.
            class NGRAPH_API ScatterND : public op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"ScatterND", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ScatterND() = default;
                /// \param data The tensor whithn slice-values will be updated
                /// \param indices Index tensor: Data type must be `element::i32` or `element::i64`
                /// \param updates The tensor of replacement-slice-values
                ScatterND(const Output<Node>& data,
                          const Output<Node>& indices,
                          const Output<Node>& updates);

                void pre_validate_and_infer_types() override;

                virtual OutputVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }
        using v0::ScatterND;
    }
}
