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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            ///
            /// \brief      Base class for ScatterXXX operators.
            ///
            class NGRAPH_API ScatterBase : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ScatterBase", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                virtual void validate_and_infer_types() override;
                virtual bool visit_attributes(AttributeVisitor& visitor) override;

            protected:
                ScatterBase() = default;

                ///
                /// \brief      Constructs ScatterBase object.
                ///
                /// \param      inputs   The input tensor to be updated.
                /// \param      indices  The tensor with indexes which will be updated.
                /// \param      updates  The tensor with update values.
                /// \param[in]  axis     The axis at which elements will be updated.
                ///
                ScatterBase(const Output<Node>& inputs,
                            const Output<Node>& indices,
                            const Output<Node>& updates,
                            const Output<Node>& axis);

            private:
                // Respective input ordinal number.
                static constexpr int DATA = 0;
                static constexpr int INDICES = 1;
                static constexpr int UPDATES = 2;
                static constexpr int AXIS = 3;
            };
        }
    }
}
