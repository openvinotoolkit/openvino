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
        namespace v0
        {
            class NGRAPH_API ReorgYolo : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ReorgYolo", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ReorgYolo() = default;
                /// \brief Constructs a ReorgYolo operation
                ///
                /// \param input          Input
                /// \param stride         Stride to reorganize input by
                ReorgYolo(const Output<Node>& input, const size_t stride);

                // Constructor with `strides` for backward compatibility
                ReorgYolo(const Output<Node>& input, const Strides& strides);

                void validate_and_infer_types() override;

                virtual bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                Strides get_strides() const { return m_strides; }

            private:
                Strides m_strides;
            };
        }
        using v0::ReorgYolo;
    }
}
