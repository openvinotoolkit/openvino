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

#include <memory>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Concatenation operation.
            class NGRAPH_API Concat : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Constructs a concatenation operation.
                Concat() = default;
                /// \brief Constructs a concatenation operation.
                ///
                /// \param args               The outputs producing the input tensors.
                /// \param axis The axis along which to concatenate the input tensors.
                Concat(const OutputVector& args, int64_t axis);

                /// \brief Constructs a concatenation operation.
                ///
                /// \param args               The nodes producing the input tensors.
                /// \param axis The axis along which to concatenate the input tensors.
                Concat(const NodeVector& args, int64_t axis);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The concatenation axis.
                int64_t get_concatenation_axis() const { return m_concat_axis; }
                void set_concatenation_axis(int64_t concatenation_axis)
                {
                    m_concat_axis = concatenation_axis;
                }
                /// \return The concatenation axis.
                int64_t get_axis() const { return m_axis; }
                void set_axis(int64_t axis) { m_axis = axis; }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool evaluate_lower(const HostTensorVector& output_values) const override;
                bool evaluate_upper(const HostTensorVector& output_values) const override;

            protected:
                /// \ brief m_axis stores default value for all iterations
                int64_t m_axis;
                /// \brief m_concat_axis stores m_axis plus the number of rank for each iteration
                int64_t m_concat_axis = -1;
            };
        }
        using v0::Concat;
    }
}
