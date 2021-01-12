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

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/op/util/fused_op.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief  Global Response Normalization with L2 norm (across channels only).
            ///
            class NGRAPH_API GRN : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"GRN", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                GRN();
                /// \brief      Constructs a GRN operation.
                ///
                /// \param      data  - Node producing the input tensor
                /// \param      bias  - The bias added to the variance.
                ///
                GRN(const Output<Node>& data, float bias);

                bool visit_attributes(AttributeVisitor& visitor) override;
                float get_bias() const { return m_bias; }
                virtual void pre_validate_and_infer_types() override;
                virtual OutputVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            protected:
                float m_bias = 1.0f;
            };
        }
        using v0::GRN;
    }
}

NGRAPH_SUPPRESS_DEPRECATED_END
