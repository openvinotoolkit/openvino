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
            /// \brief Operator performing Scale Shift transformation.
            ///
            /// Y = Scale * Data + Shift
            ///
            class NGRAPH_API ScaleShift : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"ScaleShift", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ScaleShift() = default;
                /// \brief Constructs an ScaleShift operation.
                ///
                /// \param data Input tensor
                /// \param scale Input tensor that scale input data
                /// \param shift Input tensor that shift input data
                ScaleShift(const Output<Node>& data,
                           const Output<Node>& scale,
                           const Output<Node>& shift);

                virtual OutputVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }
        using v0::ScaleShift;
    }
}
