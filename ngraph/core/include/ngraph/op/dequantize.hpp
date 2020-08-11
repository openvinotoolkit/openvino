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

#include "ngraph/axis_set.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Dequantize operation
            ///        Maps quantized input (q) to real output (r) using scale (s) and zero point
            ///        (z):
            ///        r = (q - o) * s
            class NGRAPH_API Dequantize : public ngraph::op::Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Dequantize", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a Dequantize operation
                Dequantize() = default;

                /// \brief Constructs a Dequantize operation
                /// \param input quantized input
                /// \param scale scale used for mapping
                /// \param zero_point zero point used for mapping
                /// \param type output element type
                /// \param axes axis positions on which `scale` and `zero_point` are specified
                Dequantize(const Output<Node>& input,
                           const Output<Node>& scale,
                           const Output<Node>& zero_point,
                           const element::Type& type,
                           const AxisSet& axes);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const AxisSet& get_axes() const { return m_axes; }
                void set_axes(const AxisSet& axes) { m_axes = axes; }
                const element::Type& get_type() const { return m_type; }
                void set_type(const element::Type& type) { m_type = type; }
            private:
                element::Type m_type;
                AxisSet m_axes;
            };
        }
        using v0::Dequantize;
    }
}
