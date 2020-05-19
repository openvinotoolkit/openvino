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
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief  Normalization input tensor with L2 norm.
            ///
            class NGRAPH_API NormalizeL2 : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"NormalizeL2", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                NormalizeL2() = default;
                ///
                /// \brief      Constructs a Normalize operation.
                ///
                /// \param      data            - Node producing the input tensor
                /// \param      axes            - Node indicating axes along which reduction is
                ///                               calculated
                /// \param      eps             - The epsilon added to L2 norm.
                /// \param      eps_mode        - Specifies how eps is combined with L2 value
                /// calculated
                ///                               before division
                ///
                NormalizeL2(const Output<Node>& data,
                            const Output<Node>& axes,
                            float eps,
                            EpsMode eps_mode);

                bool visit_attributes(AttributeVisitor& visitor) override;
                float get_eps() const { return m_eps; }
                EpsMode get_eps_mode() const { return m_eps_mode; }
                virtual NodeVector decompose_op() const override;
                virtual void pre_validate_and_infer_types() override;
                AxisSet get_reduction_axes() const;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            protected:
                float m_eps;
                EpsMode m_eps_mode;
            };
        }
        using v0::NormalizeL2;
    }
}
