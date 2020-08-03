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
        namespace v0
        {
            class NGRAPH_API QuantizedDot : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"QuantizedDot", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                QuantizedDot() = default;
                /// \brief Constructs a quantized convolution operation.
                ///
                /// \param input0 The node producing the input data batch tensor.
                /// \param input1 The node producing the filters tensor.
                /// \param input0_scale Scale to transform the input
                /// \param input0_zero_point Zero point used for mapping
                /// \param input1_scale Scale to transform the filters
                /// \param input1_zero_point Zero point used for mapping
                /// \param output_scale Scale to transform the output
                /// \param output_zero_point Zero point used for mapping
                /// \param output_type Output element type
                /// \param input0_axes Input0 axes set for channel wise quantization
                /// \param input1_axes Input1 axes set for channel wise quantization
                /// \param output_axes Output axes set for channel wise quantization
                QuantizedDot(const Output<Node>& input0,
                             const Output<Node>& input1,
                             size_t reduction_axes_count,
                             const Output<Node>& input0_scale,
                             const Output<Node>& input0_zero_point,
                             const Output<Node>& input1_scale,
                             const Output<Node>& input1_zero_point,
                             const Output<Node>& output_scale,
                             const Output<Node>& output_zero_point,
                             const element::Type& output_type,
                             const AxisSet& input0_axes = ngraph::AxisSet{},
                             const AxisSet& input1_axes = ngraph::AxisSet{},
                             const AxisSet& output_axes = ngraph::AxisSet{});

                std::shared_ptr<Node> get_input0() { return input_value(0).get_node_shared_ptr(); }
                std::shared_ptr<Node> get_input1() { return input_value(1).get_node_shared_ptr(); }
                const ngraph::element::Type& get_output_type() const { return m_output_type; }
                const ngraph::AxisSet& get_input0_axes() const { return m_input0_axes; }
                const ngraph::AxisSet& get_input1_axes() const { return m_input1_axes; }
                const ngraph::AxisSet& get_output_axes() const { return m_output_axes; }
                size_t get_reduction_axes_count() const { return m_reduction_axes_count; }
                void set_reduction_axes_count(size_t reduction_axes_count)
                {
                    m_reduction_axes_count = reduction_axes_count;
                }
                void validate_and_infer_types() override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            protected:
                size_t m_reduction_axes_count;
                bool m_has_reduction_axes_count;
                ngraph::element::Type m_output_type;
                ngraph::AxisSet m_input0_axes;
                ngraph::AxisSet m_input1_axes;
                ngraph::AxisSet m_output_axes;
            };
        }
        using v0::QuantizedDot;
    }
}
