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
#include "ngraph/op/util/fused_op.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Mod returns an element-wise division reminder with two given tensors applying
            /// multi-directional broadcast rules.
            class NGRAPH_API Mod : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"Mod", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Mod();
                /// \brief Constructs a Mod node.
                ///
                /// \param A - Dividend tensor
                /// \param B - Divisor tensor
                /// \param auto_broadcast Auto broadcast specification
                Mod(const Output<Node>& A,
                    const Output<Node>& B,
                    const AutoBroadcastSpec& auto_broadcast = AutoBroadcastType::NUMPY);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual OutputVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const AutoBroadcastSpec& get_auto_broadcast() const { return m_auto_broadcast; }

            private:
                AutoBroadcastSpec m_auto_broadcast;
            };
        }
    }
}

NGRAPH_SUPPRESS_DEPRECATED_END
