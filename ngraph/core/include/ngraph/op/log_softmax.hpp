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
            class NGRAPH_API LogSoftmax : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                LogSoftmax() = default;
                /// \brief Constructs a LogSoftmax operation.
                ///
                /// \param arg Node that produces the first input tensor.<br>
                /// `[d0, ...]`
                /// \param axis The axis position (0-based) on which to calculate the LogSoftmax.
                ///
                /// Output `[d0, ...]`
                ///
                LogSoftmax(const Output<Node>& arg, const int64_t axis);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                size_t get_version() const override { return 1; }
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                int64_t get_axis() const { return m_axis; }
                void set_axis(const int64_t axis) { m_axis = axis; }

            private:
                int64_t m_axis = 1;
            };
        } // namespace v5
    }     // namespace op
} // namespace ngraph
