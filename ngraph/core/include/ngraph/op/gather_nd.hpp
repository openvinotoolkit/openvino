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
        namespace v5
        {
            /// \brief GatherND operation
            ///
            class NGRAPH_API GatherND : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                GatherND() = default;

                /// \brief Constructs a GatherND operation.
                ///
                /// \param data Node producing data that are gathered
                /// \param indices Node producing indices by which the operation gathers elements
                /// or slices from data
                /// \param batch_dims Specifies a number of batch dimensions
                GatherND(const Output<Node>& data,
                         const Output<Node>& indices,
                         const size_t batch_dims = 0);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                size_t get_batch_dims() const { return m_batch_dims; }

            private:
                size_t m_batch_dims;
            };
        }
    }
}
