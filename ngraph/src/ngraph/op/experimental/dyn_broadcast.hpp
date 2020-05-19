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
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Operation which "adds" axes to an input tensor, replicating elements from the
            ///        input as needed along the new axes.
            ///
            /// This is basically the "dynamic shape" version of the static Broadcast op.
            class NGRAPH_API DynBroadcast : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"DynBroadcast", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                DynBroadcast() = default;
                /// \brief Constructs a dynamic broadcast operation.
                ///
                /// \param arg            Node that produces the input tensor to be broadcast.
                /// \param shape          Node that produces shape of the output tensor.
                /// \param broadcast_axes Node that produces the axis positions (0-based) in the
                /// result
                ///                       that are being broadcast. The remaining axes in shape must
                ///                       be
                ///                       the same as the shape of arg.
                DynBroadcast(const Output<Node>& arg,
                             const Output<Node>& shape,
                             const Output<Node>& broadcast_axes);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
            };
        }
        using v0::DynBroadcast;
    }
}
