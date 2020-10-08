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

namespace ngraph
{
    namespace op
    {
        namespace v5
        {
            /// \brief Elementwise round operation. The output is round to the nearest integer
            /// for each value. In case of halfs, the rule is defined in attribute 'mode':
            ///     'half_to_even' - round halfs to the nearest even integer.
            ///     'half_avay_from_zero': - round in such a way that the result heads away from 
            ///zero.

            class NGRAPH_API Round : public ngraph::op::Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Constructs a round operation.
                Round() = default;

                /// \brief Constructs a round operation.
                ///
                /// \param arg Node that produces the input tensor.
                /// \param mode Rule to resolve halfs
                Round(const Output<Node>& arg, const std::string& mode);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;

                std::string get_mode() const { return m_mode; }

                void set_mode(const std::string& mode) { m_mode = mode; }

            private:
                std::string m_mode;
            };
        }
    }
}
