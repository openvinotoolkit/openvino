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

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API QuantizedConvolution : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"QuantizedConvolution", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a quantized convolution operation.
                ///
                /// \param input The node producing the input data batch tensor.
                /// \param filters The node producing the filters tensor.
                /// \param window_movement_strides The window movement strides.
                /// \param window_dilation_strides The window dilation strides.
                /// \param padding_below The padding-below sizes.
                /// \param padding_above The padding-above sizes.
                /// \param data_dilation_strides The data dilation strides.
                /// \param input_scale Scale to transform the input
                /// \param input_zero_point Zero point used for mapping
                /// \param filter_scale Scale to transform the filters
                /// \param filter_zero_point Zero point used for mapping
                /// \param output_scale Scale to transform the output
                /// \param output_zero_point Zero point used for mapping
                /// \param output_type Output element type
                /// \param input_axes Input axes set for channel wise quantization
                /// \param filter_axes Filter axes set for channel wise quantization
                /// \param output_axes Output axes set for channel wise quantization
                QuantizedConvolution(const Output<Node>& input,
                                     const Output<Node>& filters,
                                     const Strides& window_movement_strides,
                                     const Strides& window_dilation_strides,
                                     const CoordinateDiff& padding_below,
                                     const CoordinateDiff& padding_above,
                                     const Strides& data_dilation_strides,
                                     const Output<Node>& input_scale,
                                     const Output<Node>& input_zero_point,
                                     const Output<Node>& filter_scale,
                                     const Output<Node>& filter_zero_point,
                                     const Output<Node>& output_scale,
                                     const Output<Node>& output_zero_point,
                                     const ngraph::element::Type& output_type,
                                     const ngraph::AxisSet& input_axes = ngraph::AxisSet{},
                                     const ngraph::AxisSet& filter_axes = ngraph::AxisSet{},
                                     const ngraph::AxisSet& output_axes = ngraph::AxisSet{});

                QuantizedConvolution() = default;

                const Strides& get_window_movement_strides() const
                {
                    return m_window_movement_strides;
                }
                const Strides& get_window_dilation_strides() const
                {
                    return m_window_dilation_strides;
                }
                const CoordinateDiff& get_padding_below() const { return m_padding_below; }
                const CoordinateDiff& get_padding_above() const { return m_padding_above; }
                const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
                std::shared_ptr<Node> get_filters() { return get_argument(1); }
                std::shared_ptr<Node> get_data_batch() { return get_argument(0); }
                const ngraph::element::Type& get_output_type() const { return m_output_type; }
                const ngraph::AxisSet& get_input_axes() const { return m_input_axes; }
                const ngraph::AxisSet& get_filter_axes() const { return m_filter_axes; }
                const ngraph::AxisSet& get_output_axes() const { return m_output_axes; }
                void validate_and_infer_types() override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

            protected:
                Strides m_window_movement_strides;
                Strides m_window_dilation_strides;
                CoordinateDiff m_padding_below;
                CoordinateDiff m_padding_above;
                Strides m_data_dilation_strides;
                ngraph::element::Type m_output_type;
                ngraph::AxisSet m_input_axes;
                ngraph::AxisSet m_filter_axes;
                ngraph::AxisSet m_output_axes;
            };
        }
        using v0::QuantizedConvolution;
    }
}
