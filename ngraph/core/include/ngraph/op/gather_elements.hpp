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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v6
        {
            /// \brief GatherElements operation
            ///
            class NGRAPH_API GatherElements : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                GatherElements() = default;

                /// \brief Constructs a GatherElements operation.
                ///
                /// \param data Node producing data that are gathered
                /// \param indices Node producing indices by which the operation gathers elements
                /// \param axis specifies axis along which indices are specified
                GatherElements(const Output<Node>& data,
                               const Output<Node>& indices,
                               const int64_t axis);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                int64_t get_axis() const { return m_axis; }

            private:
                int64_t m_axis;
            };
        }
    }
}
