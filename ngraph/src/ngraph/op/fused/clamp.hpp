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
            /// \brief Performs a clipping operation on all elements of the input node
            ///
            /// All input values that are outside of the <min;max> range are set to 'min' or 'max'
            /// depending on which side of the <min;max> range they are. The values that fall into
            /// this range remain unchanged.
            class NGRAPH_API Clamp : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"Clamp", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Clamp() = default;
                /// \brief Constructs a Clamp node.
                ///
                /// \param data - Node producing the input tensor
                /// \param min - the lower bound of the <min;max> range
                /// \param max - the upper bound of the <min;max> range
                Clamp(const Output<Node>& data, const double min, const double max);

                void pre_validate_and_infer_types() override;

                virtual OutputVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool visit_attributes(AttributeVisitor& visitor) override;

                double get_min() const { return m_min; }
                double get_max() const { return m_max; }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;

            private:
                double m_min;
                double m_max;
            };
        }
        using v0::Clamp;
    }
}
